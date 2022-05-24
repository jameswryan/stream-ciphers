//! The ChaCha20 core function. Defined in RFC 8439 Section 2.3.
//!
//! <https://tools.ietf.org/html/rfc8439#section-2.3>
//!
//! Port of x86 sse2 backend to NEON instructions

use crate::{Block, StreamClosure, Unsigned, STATE_WORDS};
use cipher::{
    consts::{U1, U64},
    BlockSizeUser, ParBlocksSizeUser, StreamBackend,
};
use core::marker::PhantomData;

use core::simd::*;

#[inline]
pub(crate) unsafe fn inner<R, F>(state: &mut [u32; STATE_WORDS], f: F)
where
    R: Unsigned,
    F: StreamClosure<BlockSize = U64>,
{
    let state_ptr = state.as_ptr() as *const [u32; 4];
    let v0 = u32x4::from_array(*state_ptr.add(0));
    let v1 = u32x4::from_array(*state_ptr.add(1));
    let v2 = u32x4::from_array(*state_ptr.add(2));
    let v3 = u32x4::from_array(*state_ptr.add(3));
    let mut backend = Backend::<R> {
        v: [v0, v1, v2, v3],
        _pd: PhantomData,
    };

    f.call(&mut backend);
    state[12] = backend.v[3][0];
}

pub(crate) struct Backend<R: Unsigned> {
    v: [u32x4; 4],
    _pd: PhantomData<R>,
}

impl<R: Unsigned> BlockSizeUser for Backend<R> {
    type BlockSize = U64;
}

impl<R: Unsigned> ParBlocksSizeUser for Backend<R> {
    type ParBlocksSize = U1;
}

impl<R: Unsigned> StreamBackend for Backend<R> {
    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        unsafe {
            // Run rounds
            let res = rounds::<R>(&self.v);

            // Increment counter
            self.v[3][0] += 1;

            // Copy res into block
            let block_ptr = block.as_mut_ptr() as *mut [u32; 4];
            for i in 0..4 {
                *block_ptr.add(i) = res[i].to_array();
            }
        }
    }
}

#[inline]
unsafe fn rounds<R: Unsigned>(v: &[u32x4; 4]) -> [u32x4; 4] {
    let mut res = *v;
    for _ in 0..R::USIZE {
        double_quarter_round(&mut res);
    }

    for i in 0..4 {
        res[i] += v[i];
    }
    res
}

#[inline]
unsafe fn double_quarter_round(v: &mut [u32x4; 4]) {
    // Column round
    add_xor_rot(v);

    // Diagonal round
    shuffle(v);
    add_xor_rot(v);
    deshuffle(v);
}

/// The goal of this function is to transform the state words from:
/// ```text
/// [a0, a1, a2, a3]    [ 0,  1,  2,  3]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c0, c1, c2, c3]    [ 8,  9, 10, 11]
/// [d0, d1, d2, d3]    [12, 13, 14, 15]
/// ```
///
/// to:
/// ```text
/// [a0, a1, a2, a3]    [ 0,  1,  2,  3]
/// [b1, b2, b3, b0] == [ 5,  6,  7,  4]
/// [c2, c3, c0, c1]    [10, 11,  8,  9]
/// [d3, d0, d1, d2]    [15, 12, 13, 14]
/// ```
///
/// so that we can apply [`add_xor_rot`] to the resulting columns, and have it compute the
/// "diagonal rounds" (as defined in RFC 7539) in parallel. In practice, this shuffle is
/// non-optimal: the last state word to be altered in `add_xor_rot` is `b`, so the shuffle
/// blocks on the result of `b` being calculated.
///
/// We can optimize this by observing that the four quarter rounds in `add_xor_rot` are
/// data-independent: they only access a single column of the state, and thus the order of
/// the columns does not matter. We therefore instead shuffle the other three state words,
/// to obtain the following equivalent layout:
/// ```text
/// [a3, a0, a1, a2]    [ 3,  0,  1,  2]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c1, c2, c3, c0]    [ 9, 10, 11,  8]
/// [d2, d3, d0, d1]    [14, 15, 12, 13]
/// ```
///
/// See https://github.com/sneves/blake2-avx2/pull/4 for additional details. The earliest
/// known occurrence of this optimization is in floodyberry's SSE4 ChaCha code from 2014:
/// - https://github.com/floodyberry/chacha-opt/blob/0ab65cb99f5016633b652edebaf3691ceb4ff753/chacha_blocks_ssse3-64.S#L639-L643
///
#[inline]
unsafe fn shuffle([a, _, c, d]: &mut [u32x4; 4]) {
    // c >>>= 32;
    *c = c.rotate_lanes_right::<3>();
    // d >>>= 64;
    *d = d.rotate_lanes_right::<2>();
    // a >>>= 96;
    *a = a.rotate_lanes_right::<1>();
}

/// Undo the shuffle operation.
/// Transform the state words from
/// ```text
/// [a3, a0, a1, a2]    [ 3,  0,  1,  2]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c1, c2, c3, c0]    [ 9, 10, 11,  8]
/// [d2, d3, d0, d1]    [14, 15, 12, 13]
/// ```
///
/// to
/// ```text
/// [a0, a1, a2, a3]    [ 0,  1,  2,  3]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c0, c1, c2, c3]    [ 8,  9, 10, 11]
/// [d0, d1, d2, d3]    [12, 13, 14, 15]
/// ```
#[inline]
unsafe fn deshuffle([a, _, c, d]: &mut [u32x4; 4]) {
    // c <<<= 32;
    *c = c.rotate_lanes_left::<3>();
    // d <<<= 64;
    *d = d.rotate_lanes_left::<2>();
    // a <<<= 96;
    *a = a.rotate_lanes_left::<1>();
}

#[inline]
unsafe fn add_xor_rot([a, b, c, d]: &mut [u32x4; 4]) {
    // a += b; d ^= a; d <<<= (16, 16, 16, 16);
    *a += *b;
    *d ^= *a;
    for i in 0..4 {
        d[i] = d[i].rotate_left(16);
    }

    // c += d; b ^= c; b <<<= (12, 12, 12, 12);
    *c += *d;
    *b ^= *c;
    for i in 0..4 {
        b[i] = b[i].rotate_left(12);
    }

    // a += b; d ^= a; d <<<= (8, 8, 8, 8);
    *a += *b;
    *d ^= *a;
    for i in 0..4 {
        d[i] = d[i].rotate_left(8);
    }

    // c += d; b ^= c; b <<<= (7, 7, 7, 7);
    *c += *d;
    *b ^= *c;
    for i in 0..4 {
        b[i] = b[i].rotate_left(7);
    }
}
