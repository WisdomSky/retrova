"""APFS engine: dissect.apfs plus the T5 Rescue fixes, battle-tested on a failing 500 GB SSD.

Patches (applied once, process-wide):
  * FileStream._read reads only the part of an extent that is asked for (stock code
    re-reads the whole extent per call - unusable for multi-GB videos) and zero-fills
    sparse extents.
  * INode.open uses the inode's private_id (data-stream id) for extents, falling back
    to the inode id - files whose data stream was swapped in (safe-save, some package
    managers) are otherwise unreadable.
  * dissect.cstruct Expression.evaluate keeps its working stack on the shared object,
    so parallel threads corrupt each other ("Invalid expression"): serialised.
"""
from __future__ import annotations

import collections
import functools
import stat
import struct
import threading

_PATCHED = False
_PATCH_LOCK = threading.Lock()


def patch_dissect():
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        _patch_cstruct()
        for patch in (_patch_apfs, _patch_btree_nodes, _patch_apfs_btree):
            try:
                patch()
            except ImportError:
                pass
        _PATCHED = True


def _patch_cstruct():
    try:
        from dissect.cstruct import expression as _cexpr
    except ImportError:
        return
    if getattr(_cexpr.Expression.evaluate, "_retrova", False):
        return
    orig_eval = _cexpr.Expression.evaluate
    eval_lock = threading.Lock()

    def locked_eval(self, cs, context=None):
        with eval_lock:
            return orig_eval(self, cs, context)

    locked_eval._retrova = True
    _cexpr.Expression.evaluate = locked_eval


def _patch_apfs():
    from io import BytesIO

    from dissect.apfs import stream as _apfs_stream
    from dissect.apfs.c_apfs import c_apfs as _c
    from dissect.apfs.objects import fs as _fs
    from dissect.util.stream import BufferedStream

    if getattr(_apfs_stream.FileStream._read, "_retrova", False):
        return

    def file_stream_read(self, offset, length):
        vol = self.volume
        bs = vol.container.block_size
        cipher = _volume_cipher(vol)
        out = []
        while length > 0:
            try:
                logical, phys, ext_len, crypto_id = self._lookup(offset)
            except _apfs_stream.Error:
                alt = getattr(self, "_t5r_alt_oid", None)
                if alt is None:
                    raise
                # try the other identifier once, then stick with whichever works
                self.oid, self._t5r_alt_oid = alt, None
                logical, phys, ext_len, crypto_id = self._lookup(offset)
            off_in = offset - logical
            avail = ext_len - off_in
            if avail <= 0:
                raise _apfs_stream.Error(f"bad extent for oid {self.oid} at {offset}")
            n = min(length, avail)
            if phys == 0:
                data = bytes(n)
            else:
                skip = off_in % bs
                nblocks = (skip + n + bs - 1) // bs
                data = vol.container._read_block(phys + off_in // bs, nblocks)
                if cipher is not None:
                    if len(data) != nblocks * bs:
                        raise _apfs_stream.Error(f"short read for oid {self.oid} at {offset}")
                    # AES-XTS: the tweak counts 512-byte sectors from the extent's crypto id
                    data = cipher.decrypt(data, crypto_id * vol.container.sectors_per_block
                                          + (off_in // bs) * vol.container.sectors_per_block)
                data = data[skip:skip + n]
                if len(data) < n:
                    raise _apfs_stream.Error(f"short read for oid {self.oid} at {offset}")
            out.append(data)
            offset += n
            length -= n
        return b"".join(out)

    file_stream_read._retrova = True
    _apfs_stream.FileStream._read = file_stream_read

    def inode_open(self):
        if self.is_compressed():
            return _apfs_stream.DecmpfsStream(self)
        if dstream := self.xfields.get(_c.INO_EXT_TYPE.DSTREAM):
            _, dstream = dstream
            pid = self.private_id or self.oid
            fs_ = _apfs_stream.FileStream(self.volume, pid, dstream.size)
            if pid != self.oid:
                fs_._t5r_alt_oid = self.oid
            return fs_
        return BufferedStream(BytesIO(b""), size=0)

    _fs.INode.open = inode_open

    def inode_size(self):
        """Stock size() looks up the xattr B-tree for every file, even when the only xattr it
        wants (com.apple.decmpfs) can't be there. On a folder with thousands of files that
        doubled the B-tree work; ask for xattrs only when the inode says it is compressed (or
        when it has no data stream at all)."""
        if _c.INODE.HAS_UNCOMPRESSED_SIZE in self.internal_flags:
            return self.inode.uncompressed_size
        if self.is_compressed():
            if decmpfs := self.xattr.get("com.apple.decmpfs"):
                return _c.decmpfs_header(decmpfs.open()).uncompressed_size
        if dstream := self.xfields.get(_c.INO_EXT_TYPE.DSTREAM):
            return dstream[1].size
        if decmpfs := self.xattr.get("com.apple.decmpfs"):
            return _c.decmpfs_header(decmpfs.open()).uncompressed_size
        return 0

    _fs.INode.size = property(inode_size)


def _patch_btree_nodes():
    """Reuse parsed B-tree nodes inside one APFS container.

    dissect builds a fresh BTreeNode for every search, so listing a folder parses the same
    directory leaf pages again for every child inode. Keeping the (immutable, read-only) node
    objects makes their key table and cached searches reusable and cuts large-folder listings
    by roughly half. Bounded so a huge disk can't grow it without limit."""
    from dissect.apfs.objects import btree_node as _bn

    if getattr(_bn.BTreeNode.from_address.__func__, "_retrova", False):
        return
    original = _bn.BTreeNode.from_address.__func__

    def from_address(cls, container, address, count=1, *, cipher=None):
        if cls is not _bn.BTreeNode or count != 1:
            return original(cls, container, address, count, cipher=cipher)
        cache = getattr(container, "_retrova_btree_nodes", None)
        if cache is None:
            cache = container._retrova_btree_nodes = collections.OrderedDict()
        key = (address, id(cipher))
        node = cache.get(key)
        if node is not None:
            cache.move_to_end(key)
            return node
        node = original(cls, container, address, count, cipher=cipher)
        cache[key] = node
        if len(cache) > 8192:
            cache.popitem(last=False)
        return node

    from_address._retrova = True
    _bn.BTreeNode.from_address = classmethod(from_address)


def _patch_apfs_btree():
    """Make APFS B-tree lookups fast and safe on a live volume.

    Three separate problems show up when browsing a real, mounted APFS SSD:

    * dissect reads B-tree keys/values through cstruct structure constructors. That is several
      hundred thousand Python calls per folder walk; decoding the two little-endian tables with
      `struct` is much cheaper.
    * `ObjectMap.lookup()` searches with xid=0 and caches only 128 results. On a volume with
      several checkpoint versions of an object it can resolve to an *old* physical block; a B-tree
      child pointer then resolves back to the node it came from and `Cursor.search()` loops
      forever - the folder, size scan or app appears frozen. Use the newest version, cache more,
      and refuse self-referential child pointers.
    * The same corrupt/stale pointer can point outside a node's key range; fail with KeyError so
      the caller reports an unreadable entry instead of spinning.
    """
    from dissect.apfs import util as _util
    from dissect.apfs.c_apfs import c_apfs as _c
    from dissect.apfs.objects import btree as _bt, fs as _fs, omap as _omap
    from dissect.apfs.util import cmp_omap

    # ---------------------------------------------------------------- key decoding
    if not getattr(_util.parse_fs_object_key, "_retrova", False):
        qword = struct.Struct("<Q")
        types = {int(t): t for t in _c.APFS_TYPE}
        id_mask, type_mask = int(_c.OBJ_ID_MASK), int(_c.OBJ_TYPE_MASK)
        type_shift = int(_c.OBJ_TYPE_SHIFT)

        def parse_fs_object_key(data):
            (value,) = qword.unpack_from(data)
            typ = (value & type_mask) >> type_shift
            return value & id_mask, types.get(typ) or _c.APFS_TYPE(typ)

        parse_fs_object_key._retrova = True
        _util.parse_fs_object_key = parse_fs_object_key
        _fs.parse_fs_object_key = parse_fs_object_key  # fs.py imported it by name

    if not getattr(_bt.BTree._read_key, "_retrova", False):
        pair = struct.Struct("<HH")
        word = struct.Struct("<H")
        info_len = len(_c.btree_info)

        def read_key(self, node, idx):
            toc = node._data_offset + node.object.btn_table_space.off
            if node._has_fixed_kv_size:
                (offset,) = word.unpack_from(node.block, toc + idx * 4)
                size = self._key_size
            else:
                offset, size = pair.unpack_from(node.block, toc + idx * 8)
            start = node._key_offset + offset
            return bytes(node.block[start:start + size])

        def read_value(self, node, idx):
            toc = node._data_offset + node.object.btn_table_space.off
            if node._has_fixed_kv_size:
                (offset,) = word.unpack_from(node.block, toc + idx * 4 + 2)
                size = self._val_size
            else:
                offset, size = pair.unpack_from(node.block, toc + idx * 8 + 4)
            end = self._node_size - (info_len if node.is_root else 0) - offset
            return bytes(node.block[end:end + size])

        read_key._retrova = read_value._retrova = True
        _bt.BTree._read_key = read_key
        _bt.BTree._read_value = read_value

    # ---------------------------------------------------------------- child pointers
    if not getattr(_bt.BTree._node_child, "_retrova", False):
        original_child = _bt.BTree._node_child

        def node_child(self, node, idx, omap=None, oid=0, xid=None):
            if idx < 0 or idx >= node.nkeys:
                raise KeyError(f"B-tree child index {idx} out of range")
            child = original_child(self, node, idx, omap, oid, xid)
            if child.address == node.address or child.level >= node.level:
                raise KeyError(f"Invalid B-tree child pointer at block {node.address}")
            return child

        node_child._retrova = True
        _bt.BTree._node_child = node_child

    # ---------------------------------------------------------------- object map
    if not getattr(_omap.ObjectMap.__init__, "_retrova", False):
        qwords = struct.Struct("<QQ")
        newest = 0xFFFFFFFFFFFFFFFF
        deleted_flag = int(_c.OMAP_VAL_DELETED)

        def omap_lookup(self, oid, xid=None):
            node = self.btree.root
            key = (oid, newest)
            while node.is_nonleaf:
                idx = self.btree._node_search(node, key, cmp=cmp_omap)
                if idx < 0 or idx >= node.nkeys:
                    raise KeyError(f"OID {oid} not found in object map")
                node = self.btree._node_child(node, idx)
            idx = self.btree._node_search(node, key, exact=False, cmp=cmp_omap)
            if idx < 0 or idx >= node.nkeys:
                raise KeyError(f"OID {oid} not found in object map")
            found_oid, _ = qwords.unpack(self.btree._read_key(node, idx))
            if found_oid != oid:
                raise KeyError(f"OID {oid} not found in object map")
            val = _c.omap_val(self.btree._read_value(node, idx))
            if val.ov_flags & deleted_flag:
                raise KeyError(f"OID {oid} is deleted in object map")
            return val.ov_paddr

        original_init = _omap.ObjectMap.__init__

        def omap_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.lookup = functools.lru_cache(16384)(omap_lookup.__get__(self, _omap.ObjectMap))

        omap_init._retrova = True
        _omap.ObjectMap.__init__ = omap_init


def _volume_cipher(vol):
    """The unlocked volume's AES-XTS cipher; None for plain volumes. Locked or per-file-key volumes raise."""
    from dissect.apfs.exception import Error
    if not vol.is_encrypted:
        return None
    if vol._cipher is None:
        raise Error("Volume is encrypted, unlock it first")
    if not vol.is_onekey:
        raise Error("Multi-key encryption is not supported")
    return vol._cipher


def kind_of(ino) -> str:
    m = ino.mode
    if stat.S_ISDIR(m):
        return "dir"
    if stat.S_ISREG(m):
        return "file"
    if stat.S_ISLNK(m):
        return "link"
    return "other"


def dirent_kind(e) -> str:
    if e.is_dir():
        return "dir"
    if e.is_file():
        return "file"
    if e.is_symlink():
        return "link"
    return "other"


def pick_volume(vols, volname=None):
    """The volume to show: the one asked for (name or '#index'), else the user-data volume."""
    if volname:
        if volname.startswith("#") and volname[1:].isdigit():
            i = int(volname[1:])
            if 0 <= i < len(vols):
                return vols[i]
        for v in vols:
            if v.name == volname:
                return v
        raise LookupError(f"There is no volume called '{volname}' here. Volumes: "
                          + ", ".join(repr(v.name) for v in vols))

    def score(v):
        try:
            role = int(v.role)
        except Exception:
            role = 0
        # 0x40 = Data (user files of a macOS install); 0 = none (plain volumes on external drives)
        r = {0x40: 0, 0: 1}.get(role, 3)
        enc = 1 if _safe(lambda: v.is_encrypted, False) else 0
        return (enc, r)

    return sorted(vols, key=score)[0]


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


class ApfsContainer:
    """An APFS container on a read-only source with a shared LRU block cache."""

    def __init__(self, src, cache_blocks: int = 8192):
        patch_dissect()
        from dissect.apfs import APFS
        self.src = src
        self.c = APFS(src)
        bs = self.c.block_size
        cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()
        clock = threading.Lock()

        def read_block(address, count=1):
            if count == 1:
                with clock:
                    hit = cache.get(address)
                    if hit is not None:
                        cache.move_to_end(address)
                        return hit
            data = src.pread(address * bs, count * bs)
            if count == 1:
                with clock:
                    cache[address] = data
                    if len(cache) > cache_blocks:
                        cache.popitem(last=False)
            return data

        self.c._read_block = read_block

    @property
    def volumes(self):
        return self.c.volumes
