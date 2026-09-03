# Compressed image API

A plan to make a tile-compressed image as easy to read and write as an
uncompressed one.

## 1. Purpose

A caller who reads a `.fz` file must handle bytes that a caller who
reads a `.fits` file does not. A caller who writes a `.fz` file must
add a header that carries no data. A caller who unpacks a `.fz` file
cannot use the library at all. No task depends on compression. All
three come from one gap in the API.

This document names that gap, and the changes that close it.

## 2. The cause

The crate has no type that can hold a decompressed image.

[`ImageHdu`](src/hdu/image.rs) borrows its pixel bytes from the
[`FitsFile`](src/hdu/file.rs) that produced it. A decompressed image
owns its bytes, so it cannot be an `ImageHdu`. Three public items
exist to work around this:

- [`OwnedImage`](src/compression.rs) holds the decompressed bytes. It
  has 5 public methods where `ImageHdu` has 13. It has no `read_raw`,
  no `read_physical`, no `read_raw_dyn` and no `n_elements`.
- [`ImageOrOwned`](src/hdu/file.rs) is the return type of
  `FitsFile::image`. It declares no methods. A caller matches the two
  variants, then calls a different method on each.
- [`Decompressed`](src/hdu/file.rs) is the item type of
  `FitsFile::iter_decompressed`. It has the same shape and the same
  problem.

Three consequences follow.

A caller who wants typed pixels from a compressed image decodes them:

```rust
let pixels: Vec<i16> = img
    .raw_bytes()
    .as_chunks::<2>()
    .0
    .iter()
    .map(|c| i16::from_be_bytes(*c))
    .collect();
```

The same caller writes `img.read_raw::<i16>()?` for an uncompressed
image. Byte order is a storage detail. The API states it here because
`OwnedImage` lacks the accessor.

The binary carries the decoder a second time.
[`decode_owned_physical`](src/main.rs) reimplements
`ImageHdu::scaled_pixels` over an `OwnedImage`. It pushes in a loop,
which the comment at [`image.rs:170`](src/hdu/image.rs) records as
half the throughput of the `collect` form.

The Python bindings copy every decompressed image.
[`python/file.rs:1913`](src/python/file.rs) calls
`owned.raw_bytes().to_vec()`, because `OwnedImage` does not release
its buffer.

Two further gaps sit next to this one.

A compressed image is a `BINTABLE` extension, so it cannot occupy the
primary slot. A caller who writes one file with one image writes two
HDUs. The first carries no data. A caller who omits it gets
`primary HDU header is missing SIMPLE = T` from
[`writer.rs:285`](src/io/writer.rs). The message names the rule. It
does not name the remedy. `fitsy fpack` builds the stub in
[`main.rs:652`](src/main.rs), so the logic sits in the binary.

The reverse operation is in the binary as well.
[`funpack_into`](src/main.rs) holds `is_bare_stub`, its `skip_stub`
predicate and `promote_to_primary`. Together these undo the move that
`fpack` made. No library caller can reach any of them.

## 3. The changes

### 3.1 Give `ImageHdu` bytes it can own or borrow

Change the `data` field of `ImageHdu` to `Cow<'a, [u8]>`, and change
the constructor to take either form:

```rust
pub fn new(header: Header, data: impl Into<Cow<'a, [u8]>>) -> Result<Self>
```

Both `&[u8]` and `Vec<u8>` satisfy `Into<Cow<'a, [u8]>>`, so the two
call sites in [`file.rs`](src/hdu/file.rs) do not change.

Then:

- `CompressedImageHdu::as_image` returns `ImageHdu<'static>`.
- `FitsFile::image` returns `ImageHdu<'_>`.
- `FitsFile::iter_decompressed` yields `Result<Hdu<'_>>`, and maps a
  compressed image to `Hdu::Image`. It then yields the same type as
  `FitsFile::iter`.

Delete `OwnedImage`, `ImageOrOwned` and `Decompressed`.

A decompressed image gains all 13 methods of `ImageHdu`. No method
forwards to another, so the two method lists cannot come to disagree.

The per-pixel cost is unchanged. `read_raw` dereferences the `Cow`
once, before the loop, and then iterates a slice.

`read_subarray` comes with the rest, and the semantics hold.
[`FitsFile::data_bytes`](src/hdu/file.rs) reads the whole data
section on first access, so a subarray of a borrowed `ImageHdu` also
reads a region of an in-memory buffer. A tile-selective read belongs
on `CompressedImageHdu`, under a separate name. Section 8 records it.

Add `into_owned(self) -> ImageHdu<'static>`, which calls
`Cow::into_owned` on the data field and returns the header unchanged.

`OwnedImage` has no lifetime parameter, so a caller stores it in a
struct without one. `ImageHdu<'a>` has a lifetime parameter, and
`FitsFile::image` binds it to the borrow of the file. The file stays
borrowed even though a decompressed image does not read from it.
`into_owned` releases that borrow, so the caller drops the file and
keeps the image.

One item of information goes away. `Decompressed` reports which HDUs
arrived compressed, because a compressed image arrives as
`Decompressed::Image` and a plain image as
`Decompressed::Hdu(Hdu::Image)`. After the change both arrive as
`Hdu::Image`. A caller who needs the distinction iterates with
`FitsFile::iter` and matches `Hdu::CompressedImage`.

### 3.2 Write the primary stub from `write_hdu_compressed`

`FitsWriter::write_hdu_compressed` writes the stub when `hdu_count`
is 0, then writes the compressed HDU.

A compressed image can never occupy HDU 0, so the current error
protects no correct caller. The change converts an error into the
only correct action.

This deletes the `i == 0` block in `fpack_into`. Checksum stamping is
unaffected, because the stub goes through `write_hdu`.

Add no second method. A name such as `write_image_compressed` differs
from `write_hdu_compressed` by one word, and differs in behavior by a
whole HDU.

### 3.3 Add `FitsFile::write_decompressed`

Add `FitsFile::write_decompressed(path, overwrite)`, beside the
existing `FitsFile::write`. It decompresses every tile-compressed
image and writes the result.

It absorbs `is_bare_stub`, the `skip_stub` predicate and
`promote_to_primary` from `funpack_into`. Keep `is_bare_stub` private.

The stub drops under a fixed policy. Add no parameter for it. The
operation is the inverse of `fpack`, and `fpack` inserts the stub, so
the inverse removes it.

The drop discards no information, which is what makes a fixed policy
correct. `is_bare_stub` holds only when the header carries `SIMPLE`,
`BITPIX`, `NAXIS`, `EXTEND`, `CHECKSUM` and `DATASUM`, and no other
card. `Header::promote_to_primary` regenerates the first four from the
image it promotes. `CHECKSUM` and `DATASUM` cover the stub itself, so
they say nothing about the image, and the writer stamps new ones. A
stub that carries any other card fails `is_bare_stub`, and the HDU
stays.

Three further conditions apply, and all must hold: the stub is HDU 0,
its data section is empty, and HDU 1 is a compressed image with
`ZSIMPLE = T`.

State this in the documentation of `write_decompressed`. A caller who
reads the signature alone cannot tell that an HDU count can drop by
one.

`fitsy funpack` then parses its arguments and makes one call.

### 3.4 Add `Header::promote_to_primary`

Add `Header::promote_to_primary(&self) -> Result<Header>`. It rebuilds
an `IMAGE` extension header as a primary header: `SIMPLE` for
`XTENSION`, `EXTEND` after the last `NAXISn` card, and no `PCOUNT` or
`GCOUNT`.

Section 3.3 needs it. A caller who reorders HDUs needs it too.

It also shortens the comment in `synthesize_image_header`, which
records that the function cannot emit a promotable header.

Add no inverse method. No caller needs one.

### 3.5 Add `ImageHdu::into_bytes`

Add `into_bytes(self) -> Vec<u8>`, which returns `Cow::into_owned` of
the data field. The Python binding then moves the decompressed buffer
instead of copying it.

### 3.6 Report a compressed HDU from `FitsUpdater`

`FitsUpdater::write_image_subarray` reports
`HDU {i} is not an image (or out of range)` for a tile-compressed
image, at [`update.rs:288`](src/io/update.rs). The statement is false.

A rewritten tile changes size, so an in-place patch cannot work.
Return an error that says so. State the same limit in the
documentation of [`FitsUpdater`](src/io/update.rs) and
[`FitsAppender`](src/io/append.rs), which do not mention compression.

## 4. Result

```rust
let mut w = FitsWriter::new(&mut out);
w.write_hdu_compressed(&header, &data, &opts)?;
w.finish()?;

let f = FitsFile::open(&path)?;
let img = f.image(1)?;
println!("{:?} {:?}", img.bitpix(), img.axes());
let pixels = img.read_raw::<i16>()?;
println!("pixels identical: {}", pixels.as_slice() == &original[..]);
```

The same code reads a compressed image and an uncompressed one, because
one type holds both. Byte order, tile shape and the primary stub stay
inside the library.

## 5. What this removes

| Item | Reason |
| --- | --- |
| `OwnedImage` | `ImageHdu` holds a decompressed image |
| `ImageOrOwned` | `FitsFile::image` returns one type |
| `Decompressed` | `iter_decompressed` yields `Hdu` |
| `decode_owned_physical` in `main.rs` | `read_physical` covers it |
| The `i == 0` stub block in `fpack_into` | Section 3.2 |
| `is_bare_stub`, `skip_stub`, `promote_to_primary` in `main.rs` | Section 3.3 |
| The copy at `python/file.rs:1913` | Section 3.5 |

The public API loses three types and gains four methods.

## 6. Order

1. 3.1. It is the substance, and it removes the need for per-type
   accessors.
2. 3.4, then 3.3. The second needs the first.
3. 3.2.
4. 3.5 and 3.6.
5. Rewrite `examples/compress_image.rs` on the new API.

## 7. Compatibility

This release is 0.5.0. Three public items go away.

`OwnedImage` is the only one exported at the crate root. Its one use
outside `src/` is [`tests/compression_write.rs:94`](tests/compression_write.rs).

`ImageOrOwned` and `Decompressed` are reachable only as
`fitsy::hdu::file::ImageOrOwned` and `fitsy::hdu::file::Decompressed`.
Neither [`hdu.rs`](src/hdu.rs) nor [`lib.rs`](src/lib.rs) re-exports
them.

`FitsFile::image` and `FitsFile::iter_decompressed` change their
return types.

`ImageHdu::new` changes its second parameter from `&'a [u8]` to
`impl Into<Cow<'a, [u8]>>`. A caller who passes a `&[u8]` is
unaffected.

`ImageHdu` derives `Clone`, and the cost of a clone changes. A clone
of a borrowed image copies a `Header` and a slice reference. A clone
of an owned image copies the pixel buffer as well. No code in `src/`
clones an `ImageHdu`, because every internal use takes a reference, so
this reaches downstream callers alone. State the cost on the type.

One behavior change follows from 3.1. `ImageHdu::new` checks that
`data.len()` matches the size the header declares, and the
decompressed path now runs that check. `CompressedImageHdu::decompress`
returns exactly `n_pixels * byte_size` bytes, so the check passes. A
decode defect that broke that invariant now returns an error where it
previously returned wrong pixels.

Record every item in `CHANGELOG.md`.

## 8. Deferred

- **Tile-selective read.** `CompressedImageHdu::read_region(start,
  shape)` decodes only the tiles the region intersects. The default
  tile shape is one row, so a cutout from a large image reads the rows
  it asks for. This adds API surface, so it does not belong in the
  release that removes three types.
- **Reusable output buffer.** `decompress_into(&mut Vec<u8>)` lets a
  caller reuse one allocation across HDUs.
- **Streaming write.** `compress_image_to_hdu` holds the uncompressed
  image and the compressed payload at the same time. Peak memory is
  about twice the image. Keep `TileOpts` opaque so a later change can
  address this.
- **Module split.** [`compression.rs`](src/compression.rs) is 2130
  lines and handles two conventions: whole-file gzip and tile
  compression. `maybe_gunzip` belongs in `io`.
- **Parallel tiles.** Each tile decodes independently, and
  `scatter_tile` writes disjoint output. The shared scratch buffers
  are the only obstacle. This is internal, so it needs no decision now.
