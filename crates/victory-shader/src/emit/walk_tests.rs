//! Wave64 programs whose halves meet on a 32-lane host
//! ([`Options::wave32_host`]) only once the per-lane branches around their
//! cross-lane operations and LDS reads are taken by the whole console wave
//! (`widened`, `structure::structure_wide`). Synthetic GFX10 programs,
//! written for these tests. Each is refused without the walk
//! ([`Options::no_wave_walk`]: the halves cannot meet in a per-lane branch)
//! and made with it, in 32-lane waves. `native::walk_tests` in `victory-gpu`
//! runs the same programs on the GPU against the console's results.

use super::{HostShader, Options};
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::{dxbc, io, ir, resource, structure};

// Operand codes.
const VCC: u32 = 106;
const EXEC: u32 = 126;
const LITERAL: u32 = 255;
/// An inline integer constant 0..=64.
const fn k(n: u32) -> u32 {
    128 + n
}
/// VGPR `n` as a source.
const fn v(n: u32) -> u32 {
    256 + n
}

/// (`vsrc1` a VGPR, given as a source: [`v`].)
const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | (vsrc1 & 0xFF) << 9 | src0
}
const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}
/// A compare writing VCC.
const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | (vsrc1 & 0xFF) << 9 | src0
}
/// A compare writing an SGPR pair (VOP3).
const fn vop3c(op: u32, sdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | sdst, src0 | src1 << 9]
}
/// A VOP3 instruction of two sources (`v_readlane_b32`, `v_writelane_b32`).
const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}
const fn sop1(op: u32, sdst: u32, src: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src
}
const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}
/// A compare of two scalars, setting SCC.
const fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
    0xBF00_0000 | op << 16 | src1 << 8 | src0
}
/// A branch `words` dwords past the next instruction.
const fn sopp(op: u32, words: i32) -> u32 {
    0xBF80_0000 | op << 16 | (words as u32 & 0xFFFF)
}

const V_MOV: u32 = 0x01;
const V_READFIRSTLANE: u32 = 0x02;
const V_LSHRREV: u32 = 0x16;
const V_LSHLREV: u32 = 0x1A;
const V_AND: u32 = 0x1B;
const V_XOR: u32 = 0x1D;
const V_ADD: u32 = 0x25;
const V_SUBREV: u32 = 0x27;
const V_CMP_LT_U32: u32 = 0xC1;
const V_CMP_EQ_U32: u32 = 0xC2;
const V_CMP_GT_U32: u32 = 0xC4;
const V_CMP_NE_U32: u32 = 0xC5;
const V_CMPX_GT_U32: u32 = 0xD4;
const V_READLANE: u32 = 0x360;
const V_WRITELANE: u32 = 0x361;
const S_MOV_B32: u32 = 0x03;
const S_MOV_B64: u32 = 0x04;
const S_BCNT1_B64: u32 = 0x10;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_ADD_I32: u32 = 0x02;
const S_AND_B64: u32 = 0x0F;
const S_LSHR_B32: u32 = 0x20;
const S_CMP_LG_U32: u32 = 0x07;
const S_ENDPGM: u32 = 0xBF81_0000;
const SCC1: u32 = 0x05;
const VCCZ: u32 = 0x06;
const EXECZ: u32 = 0x08;
const EXECNZ: u32 = 0x09;

/// Root constants in the user SGPRs `s0..s7`; the output's title address
/// in `s[6:7]`.
pub(crate) const SGPRS: u32 = 8;

/// `global_store_dword v1, v<data>, s[6:7] offset:<offset>`: the dword at
/// byte v1 + offset of the output.
const fn store(data: u32, offset: u32) -> [u32; 2] {
    [0xDC70_8000 | offset, 0x0006_0001 | data << 8]
}
/// `ds_write_b32 v<addr>, v<data>`.
const fn ds_write(addr: u32, data: u32) -> [u32; 2] {
    [0xD834_0000, addr | data << 8]
}
/// `ds_read_b32 v<dst>, v<addr>`.
const fn ds_read(dst: u32, addr: u32) -> [u32; 2] {
    [0xD8D8_0000, addr | dst << 24]
}

/// v1: the byte of the thread's dword (4 × its id).
const PROLOGUE: [u32; 1] = [vop2(V_LSHLREV, 1, k(2), v(0))];

/// `WaveActiveSum(v16)` as the console compiler expands it: the sum in
/// s12, EXEC as it was saved in s[0:1].
const WAVE_SUM_V16_S12: [u32; 19] = [
    0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
    0xD501_0002,
    0x0002_2080, // v_cndmask_b32 v2, 0, v16, s[0:1]
    0x4A04_04E9,
    0xDE54_C102, // v_add_nc_u32 v2, v2, v2 dpp8:[1,0,3,2,5,4,7,6]
    0x4A04_04E9,
    0x9360_1202, // v_add_nc_u32 v2, v2, v2 dpp8:[2,2,0,0,6,6,4,4]
    0x4A04_04E9,
    0x0009_2402, // v_add_nc_u32 v2, v2, v2 dpp8:[4,4,4,4,0,0,0,0]
    0x4A04_04FA,
    0xFF09_4002, // v_add_nc_u32 v2, v2, v2 row_mirror bound_ctrl:1
    0xD778_0003,
    0x0201_0102, // v_permlanex16_b32 v3, v2, 0, 0
    0x4A04_0503, // v_add_nc_u32 v2, v3, v2
    0xD760_0002,
    0x0001_4102, // v_readlane_b32 s2, v2, 32
    0x4A04_0402, // v_add_nc_u32 v2, s2, v2
    0xD760_000C,
    0x0001_0102, // v_readlane_b32 s12, v2, 0
];

/// `if (lanes) { body }`: EXEC saved in s[20:21] and narrowed to VCC, the
/// skip, `body`, EXEC back.
fn branch(lanes: &[u32], body: &[u32]) -> Vec<u32> {
    let mut w = lanes.to_vec();
    w.push(sop1(S_AND_SAVEEXEC_B64, 20, VCC));
    w.push(sopp(EXECZ, body.len() as i32));
    w.extend(body);
    w.push(sop1(S_MOV_B64, EXEC, 20));
    w
}

/// A count of the even lanes below 40 of those that are even: a lane
/// mask made and read as a number inside `if (id % 2 == 0)`. Each even
/// lane stores 20, the others nothing.
fn count_in_branch() -> Vec<u32> {
    let mut body = vop3c(V_CMP_GT_U32, 14, k(40), v(0)).to_vec();
    body.push(sop1(S_BCNT1_B64, 16, 14));
    body.push(vop1(V_MOV, 2, 16));
    body.extend(store(2, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vop2(V_AND, 2, k(1), v(0)));
    w.push(vopc(V_CMP_EQ_U32, k(0), v(2)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// A loop each lane goes round `id % 4 + 1` times (its exit per lane, by
/// EXEC), adding the lanes still in it to its sum and 1 to a scalar count:
/// the console's wave goes round 4 times, 64, 48, 32 and 16 lanes, and the
/// count is the wave's. Each lane stores its sum and the count (at +256).
fn loop_by_lane() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        vop2(V_AND, 3, k(3), v(0)),
        vop2(V_ADD, 3, k(1), v(3)),
        vop1(V_MOV, 4, k(0)),
        sop1(S_MOV_B32, 17, k(0)),
        sop1(S_MOV_B64, 12, EXEC),
        // the loop
        sop1(S_BCNT1_B64, 16, EXEC),
        vop2(V_ADD, 4, 16, v(4)),
        sop2(S_ADD_I32, 17, 17, k(1)),
        vop2(V_SUBREV, 3, k(1), v(3)),
        vopc(V_CMP_NE_U32, k(0), v(3)),
        sop2(S_AND_B64, EXEC, EXEC, VCC),
        sopp(EXECNZ, -7),
        sop1(S_MOV_B64, EXEC, 12),
        vop1(V_MOV, 5, 17),
    ]);
    w.extend(store(4, 0));
    w.extend(store(5, 256));
    w.push(S_ENDPGM);
    w
}

/// [`loop_by_lane`]'s loop with, inside it, `if (id % 2 == 0)` counting the
/// lanes there (a mask made in the branch, as [`count_in_branch`]): each
/// pass the even lanes still in the loop, 32, 16, 16 and none, added up by
/// each even lane while it goes round: 32 where id % 4 is 0 (one pass), 64
/// where it is 2 (three); odd lanes add nothing.
fn branch_in_loop_by_lane() -> Vec<u32> {
    let mut count = vop3c(V_CMP_GT_U32, 14, k(64), v(0)).to_vec();
    count.push(sop1(S_BCNT1_B64, 16, 14));
    count.push(vop2(V_ADD, 4, 16, v(4)));
    let mut body = branch(&[vopc(V_CMP_EQ_U32, k(0), v(5))], &count);
    body.extend([
        vop2(V_SUBREV, 3, k(1), v(3)),
        vopc(V_CMP_NE_U32, k(0), v(3)),
        sop2(S_AND_B64, EXEC, EXEC, VCC),
    ]);
    let back = -(body.len() as i32 + 1);
    let mut w = PROLOGUE.to_vec();
    w.extend([
        vop2(V_AND, 3, k(3), v(0)),
        vop2(V_ADD, 3, k(1), v(3)),
        vop1(V_MOV, 4, k(0)),
        vop2(V_AND, 5, k(1), v(0)),
        sop1(S_MOV_B64, 12, EXEC),
    ]);
    w.extend(body);
    w.push(sopp(EXECNZ, back));
    w.push(sop1(S_MOV_B64, EXEC, 12));
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// Inside `if (id % 2 == 0)`: each lane writes its id to LDS and reads the
/// dword of lane id ^ 32, which the other half wrote (no barrier: the
/// console's wave runs in lockstep). Even lanes store id ^ 32.
fn lds_in_branch() -> Vec<u32> {
    let mut body = ds_write(1, 0).to_vec();
    body.extend([sop1(S_MOV_B32, 18, LITERAL), 128]);
    body.push(vop2(V_XOR, 6, 18, v(1)));
    body.extend(ds_read(7, 6));
    body.extend(store(7, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vop2(V_AND, 2, k(1), v(0)));
    w.push(vopc(V_CMP_EQ_U32, k(0), v(2)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// `WaveActiveSum(id)` inside `if (id < 50)`: 1225 in each such lane.
fn sum_in_branch() -> Vec<u32> {
    let mut body = vec![vop1(V_MOV, 16, v(0))];
    body.extend(WAVE_SUM_V16_S12);
    body.push(sop1(S_MOV_B64, EXEC, 0));
    body.push(vop1(V_MOV, 2, 12));
    body.extend(store(2, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vopc(V_CMP_GT_U32, k(50), v(0)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// The lanes from 56 leave (`if (id >= 56) return;`), then
/// [`count_in_branch`]'s count for the rest: 20 in each even lane below
/// 56.
fn count_after_return() -> Vec<u32> {
    let rest = count_in_branch();
    let mut w = PROLOGUE.to_vec();
    w.push(vopc(V_CMPX_GT_U32, k(56), v(0)));
    // to the end: past the rest of `count_in_branch` but its own end
    let tail = &rest[PROLOGUE.len()..];
    w.push(sopp(EXECZ, tail.len() as i32 - 1));
    w.extend(tail);
    w
}

/// [`loop_by_lane`] with each lane going round `id / 32 + 1` times: the
/// lower half leaves after one pass, the upper half goes round again
/// alone (in the second pass one host wave has no lane in the loop and
/// the other has all of its lanes). The console's wave goes round twice,
/// 64 lanes then 32: the lower half's lanes store 64, the upper half's
/// 96, and every lane the count 2 (at +256).
fn loop_by_half() -> Vec<u32> {
    let mut w = loop_by_lane();
    // v_and_b32 v3, 3, v0 -> v_lshrrev_b32 v3, 5, v0
    assert_eq!(w[PROLOGUE.len()], vop2(V_AND, 3, k(3), v(0)));
    w[PROLOGUE.len()] = vop2(V_LSHRREV, 3, k(5), v(0));
    w
}

/// `if (id >= 32)` (an `execz` skip) around a count of the lanes there
/// below 48: only the upper half has lanes in the branch, so one host
/// wave would skip it and the other not. On the console the wave goes in
/// (some lane is in EXEC): each upper lane stores 16, the lower half
/// nothing.
fn upper_half_branch() -> Vec<u32> {
    let mut body = vop3c(V_CMP_GT_U32, 14, k(48), v(0)).to_vec();
    body.push(sop1(S_BCNT1_B64, 16, 14));
    body.push(vop1(V_MOV, 2, 16));
    body.extend(store(2, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vopc(V_CMP_LT_U32, k(31), v(0)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// A `vccz` skip on `id >= 32` (VCC within EXEC), EXEC left whole: the
/// console's wave goes on when some lane's VCC bit is set, so every lane
/// runs the count of the lanes below 48 and stores 48. Only the upper
/// half has VCC bits set: one host wave alone would skip.
fn upper_half_vccz() -> Vec<u32> {
    let mut body = vop3c(V_CMP_GT_U32, 14, k(48), v(0)).to_vec();
    body.push(sop1(S_BCNT1_B64, 16, 14));
    body.push(vop1(V_MOV, 2, 16));
    body.extend(store(2, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vopc(V_CMP_LT_U32, k(31), v(0)));
    w.push(sop2(S_AND_B64, VCC, EXEC, VCC));
    w.push(sopp(VCCZ, body.len() as i32));
    w.extend(body);
    w.push(S_ENDPGM);
    w
}

/// `if (s18 >> 5 != 0) return;` before [`count_in_branch`]'s count, with
/// s18 made by `make`: a scalar branch both halves reach together, ahead
/// of where they must meet.
fn scalar_return_then_count(make: &[u32]) -> Vec<u32> {
    let rest = count_in_branch();
    let tail = &rest[PROLOGUE.len()..];
    let mut w = PROLOGUE.to_vec();
    w.extend(make);
    w.extend([
        sop2(S_LSHR_B32, 18, 18, k(5)),
        sopc(S_CMP_LG_U32, 18, k(0)),
        // to the end: past the rest of `count_in_branch` but its own end
        sopp(SCC1, tail.len() as i32 - 1),
    ]);
    w.extend(tail);
    w
}

/// [`scalar_return_then_count`] on a root constant (s0, 0 in the GPU
/// test): every lane of the console wave holds it alike, so both halves
/// go the same way. 20 in each even lane.
fn root_constant_return_then_count() -> Vec<u32> {
    scalar_return_then_count(&[sop1(S_MOV_B32, 18, 0)])
}

/// [`scalar_return_then_count`] on the first lane's id
/// (`v_readfirstlane_b32 s18, v0`): 0 on the console, so its wave goes on,
/// but on a 32-lane host each half reads its own first lane, 0 and 32, and
/// the upper half would leave while the lower half waits for it at the
/// count's exchange: the GPU would hang.
fn first_lane_return_then_count() -> Vec<u32> {
    scalar_return_then_count(&[vop1(V_READFIRSTLANE, 18, v(0))])
}

/// `s18` spilled to lane 3 of v5 and read back: what `make` left in s18.
fn spilled_and_restored(make: &[u32]) -> Vec<u32> {
    let mut w = make.to_vec();
    w.extend(vop3(V_WRITELANE, 5, 18, k(3)));
    w.extend(vop3(V_READLANE, 18, v(5), k(3)));
    w
}

/// [`first_lane_return_then_count`] with the first lane's id spilled to a
/// VGPR lane and read back before the branch: the read gives back the
/// SGPR spilled, each half's own, so the halves would part as before.
fn spilled_first_lane_return_then_count() -> Vec<u32> {
    scalar_return_then_count(&spilled_and_restored(&[vop1(V_READFIRSTLANE, 18, v(0))]))
}

/// [`root_constant_return_then_count`] with the root constant spilled to a
/// VGPR lane and read back before the branch: alike in both halves still.
fn spilled_root_constant_return_then_count() -> Vec<u32> {
    scalar_return_then_count(&spilled_and_restored(&[sop1(S_MOV_B32, 18, 0)]))
}

/// [`scalar_return_then_count`] on `v_readlane_b32 s18, v0, s18` with s18
/// the first lane's id: each half reads the lane its own first lane names
/// (0 and 32 on a 32-lane host), so each gets a different value.
fn lane_by_first_lane_return_then_count() -> Vec<u32> {
    let mut make = vec![vop1(V_READFIRSTLANE, 18, v(0))];
    make.extend(vop3(V_READLANE, 18, v(0), 18));
    scalar_return_then_count(&make)
}

/// [`scalar_return_then_count`] on `v_readlane_b32 s18, v0, 3`: a constant
/// lane of a per-lane VGPR, console lane 3 in both halves (the halves
/// exchange their lanes, `lower/halves.rs`), so both go the same way.
fn constant_lane_return_then_count() -> Vec<u32> {
    scalar_return_then_count(&vop3(V_READLANE, 18, v(0), k(3)))
}

/// What lane `id` stores at each dword `n` of the output, the console's
/// result (`None`: the lane stores nothing there).
pub(crate) type Want = fn(u32, u32) -> Option<u32>;

/// Name, program, LDS bytes, the dwords to read back, and what each holds.
pub(crate) fn programs() -> Vec<(&'static str, Vec<u32>, u32, u32, Want)> {
    vec![
        ("a count in a branch", count_in_branch(), 0, 64, |id, n| {
            (n == id && id % 2 == 0).then_some(20)
        }),
        ("a loop by lane", loop_by_lane(), 0, 128, |id, n| {
            let sums = [64, 112, 144, 160];
            if n == id {
                Some(sums[(id % 4) as usize])
            } else {
                (n == id + 64).then_some(4)
            }
        }),
        ("LDS in a branch", lds_in_branch(), 512, 64, |id, n| {
            (n == id && id % 2 == 0).then_some(id ^ 32)
        }),
        ("a sum in a branch", sum_in_branch(), 0, 64, |id, n| {
            (n == id && id < 50).then_some(1225)
        }),
        (
            "a count after a return",
            count_after_return(),
            0,
            64,
            |id, n| (n == id && id % 2 == 0 && id < 56).then_some(20),
        ),
        (
            "a branch in a loop by lane",
            branch_in_loop_by_lane(),
            0,
            64,
            |id, n| (n == id).then_some([32, 0, 64, 0][(id % 4) as usize]),
        ),
        ("a loop by half", loop_by_half(), 0, 128, |id, n| {
            if n == id {
                Some(if id < 32 { 64 } else { 96 })
            } else {
                (n == id + 64).then_some(2)
            }
        }),
        (
            "a branch only the upper half has lanes in",
            upper_half_branch(),
            0,
            64,
            |id, n| (n == id && id >= 32).then_some(16),
        ),
        (
            "a vccz skip only the upper half's VCC passes",
            upper_half_vccz(),
            0,
            64,
            |id, n| (n == id).then_some(48),
        ),
        (
            "a return on a root constant",
            root_constant_return_then_count(),
            0,
            64,
            |id, n| (n == id && id % 2 == 0).then_some(20),
        ),
    ]
}

/// A wave64 compute program of one console wave (64 threads, ids in v0)
/// whose user SGPRs `s0..s7` are root constants.
fn native(words: &[u32], lds_bytes: u32) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: SGPRS,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (lds_bytes / 512) << 15 | SGPRS << 1,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The `WaveSize` a translation declares (PSV0's lane counts; 0 for none).
fn wave_size(dxil: &[u8]) -> u32 {
    let u = |o: usize| u32::from_le_bytes(dxil[o..o + 4].try_into().unwrap());
    for i in 0..u(28) as usize {
        let o = u(32 + 4 * i) as usize;
        if &dxil[o..o + 4] == b"PSV0" {
            return u(o + 8 + 4 + 16);
        }
    }
    panic!("no PSV0")
}

/// The program translated for the native Metal host (32-lane waves), with
/// the walk or without it.
pub(crate) fn made(
    words: &[u32],
    lds_bytes: u32,
    walk: bool,
) -> Result<HostShader, crate::ShaderError> {
    let native = native(words, lds_bytes);
    let p = ir::build(&native.code, true, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("raise");
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let options = Options {
        wave32_host: true,
        zero_copy_memory: true,
        no_wave_walk: !walk,
        ..Options::default()
    };
    super::emit_with(&c, &native, &p, &st, &io, &res, options)
}

#[test]
fn the_test_programs_are_what_their_comments_say() {
    let text = |words: &[u32]| -> Vec<String> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .unwrap_or_else(|e| panic!("{words:08x?}: {e}"))
            .iter()
            .map(|i| i.text(true))
            .collect()
    };
    assert_eq!(
        text(&loop_by_lane()[PROLOGUE.len()..]),
        [
            "v_and_b32 v3, 3, v0",
            "v_add_nc_u32 v3, 1, v3",
            "v_mov_b32 v4, 0",
            "s_mov_b32 s17, 0",
            "s_mov_b64 s[12:13], exec",
            "s_bcnt1_i32_b64 s16, exec",
            "v_add_nc_u32 v4, s16, v4",
            "s_add_i32 s17, s17, 1",
            "v_subrev_nc_u32 v3, 1, v3",
            "v_cmp_ne_u32 vcc, 0, v3",
            "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz label_0014",
            "s_mov_b64 exec, s[12:13]",
            "v_mov_b32 v5, s17",
            "global_store_dword v1, v4, s[6:7]",
            "global_store_dword v1, v5, s[6:7] inst_offset:256",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(&lds_in_branch()[PROLOGUE.len()..]),
        [
            "v_and_b32 v2, 1, v0",
            "v_cmp_eq_u32 vcc, 0, v2",
            "s_and_saveexec_b64 s[20:21], vcc",
            "s_cbranch_execz label_0034",
            "ds_write_b32 v1, v0",
            "s_mov_b32 s18, lit(0x00000080)",
            "v_xor_b32 v6, s18, v1",
            "ds_read_b32 v7, v6",
            "global_store_dword v1, v7, s[6:7]",
            "s_mov_b64 exec, s[20:21]",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(&branch_in_loop_by_lane()[PROLOGUE.len() + 5..]),
        [
            "v_cmp_eq_u32 vcc, 0, v5",
            "s_and_saveexec_b64 s[20:21], vcc",
            "s_cbranch_execz label_001C",
            "v_cmp_gt_u32 s[14:15], 64, v0",
            "s_bcnt1_i32_b64 s16, s[14:15]",
            "v_add_nc_u32 v4, s16, v4",
            "s_mov_b64 exec, s[20:21]",
            "v_subrev_nc_u32 v3, 1, v3",
            "v_cmp_ne_u32 vcc, 0, v3",
            "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz label_0000",
            "s_mov_b64 exec, s[12:13]",
            "global_store_dword v1, v4, s[6:7]",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(
            &[
                &spilled_first_lane_return_then_count()[PROLOGUE.len()..][..7],
                &[S_ENDPGM]
            ]
            .concat()
        ),
        [
            "v_readfirstlane_b32 s18, v0",
            "v_writelane_b32 v5, s18, 3",
            "v_readlane_b32 s18, v5, 3",
            "s_lshr_b32 s18, s18, 5",
            "s_cmp_lg_u32 s18, 0",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(
            &[
                &lane_by_first_lane_return_then_count()[PROLOGUE.len()..][..4],
                &[S_ENDPGM]
            ]
            .concat()
        ),
        [
            "v_readfirstlane_b32 s18, v0",
            "v_readlane_b32 s18, v0, s18",
            "s_lshr_b32 s18, s18, 5",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(&count_after_return()[PROLOGUE.len()..]),
        [
            "v_cmpx_gt_u32 exec, 56, v0",
            "s_cbranch_execz label_0034",
            "v_and_b32 v2, 1, v0",
            "v_cmp_eq_u32 vcc, 0, v2",
            "s_and_saveexec_b64 s[20:21], vcc",
            "s_cbranch_execz label_0030",
            "v_cmp_gt_u32 s[14:15], 40, v0",
            "s_bcnt1_i32_b64 s16, s[14:15]",
            "v_mov_b32 v2, s16",
            "global_store_dword v1, v2, s[6:7]",
            "s_mov_b64 exec, s[20:21]",
            "s_endpgm",
        ]
    );
}

/// Each program is refused without the walk (its halves would have to meet
/// in a per-lane branch or loop), and runs in 32-lane waves with it.
#[test]
fn the_halves_meet_inside_branches_taken_wave_wide() {
    for (name, words, lds, _, _) in programs() {
        let before = made(&words, lds, false);
        assert!(before.is_err(), "{name}: made without the walk");
        let walked = made(&words, lds, true).unwrap_or_else(|e| panic!("{name}: the walk: {e}"));
        assert_eq!(wave_size(&walked.dxil), 32, "{name}: walked");
    }
}

/// Programs whose halves meet without the walk are made as before.
#[test]
fn programs_whose_halves_meet_are_made_as_before() {
    // A ballot each lane runs once, and `if (id >= 20) v2 = id;`.
    let words = [
        vop2(V_LSHLREV, 1, k(2), v(0)),
        vop3c(V_CMP_GT_U32, 14, k(40), v(0))[0],
        vop3c(V_CMP_GT_U32, 14, k(40), v(0))[1],
        sop1(S_BCNT1_B64, 16, 14),
        vop1(V_MOV, 2, 16),
        vopc(V_CMP_GT_U32, k(20), v(0)),
        sop1(S_AND_SAVEEXEC_B64, 20, VCC),
        sopp(EXECZ, 1),
        vop1(V_MOV, 2, v(0)),
        sop1(S_MOV_B64, EXEC, 20),
        store(2, 0)[0],
        store(2, 0)[1],
        S_ENDPGM,
    ];
    let with = made(&words, 0, true).expect("with the walk");
    let without = made(&words, 0, false).expect("without the walk");
    assert_eq!(with.dxil, without.dxil);
}

/// A scalar branch the halves reach together tests what each half may
/// hold differently (a first-lane read, the half's own on a 32-lane host):
/// with the walk the halves would part there and one would wait at a
/// barrier the other never reaches, so the program is refused
/// (`lower::scalar_parting`, `lower::wave_same`). The same branch on a
/// root constant, which both hold alike, is made.
#[test]
fn a_scalar_branch_on_a_lane_s_value_is_refused() {
    let first_lane = first_lane_return_then_count();
    assert!(made(&first_lane, 0, false).is_err());
    let e = made(&first_lane, 0, true).expect_err("made");
    assert!(
        e.to_string()
            .contains("a scalar branch the halves may take apart"),
        "{e}"
    );
    let root = root_constant_return_then_count();
    assert!(made(&root, 0, false).is_err());
    let l = made(&root, 0, true).expect("on a root constant");
    assert_eq!(wave_size(&l.dxil), 32);
}

/// A lane read is alike in both halves only when what it reads is: a
/// first-lane id spilled with `v_writelane_b32` and read back
/// (`Structured::restored`, the SGPR spilled) is each half's own still, and
/// a lane named by a first-lane id is a different console lane in each
/// half. Both are refused with the walk; a root constant spilled and read
/// back, and a constant lane of a per-lane VGPR, are made.
#[test]
fn a_scalar_branch_on_a_lane_read_of_a_lane_s_value_is_refused() {
    let mut made_anyway = Vec::new();
    for (name, words) in [
        (
            "a spilled first lane",
            spilled_first_lane_return_then_count(),
        ),
        (
            "a lane by the first lane",
            lane_by_first_lane_return_then_count(),
        ),
    ] {
        assert!(made(&words, 0, false).is_err(), "{name}: without the walk");
        match made(&words, 0, true) {
            Ok(_) => made_anyway.push(name),
            Err(e) => assert!(
                e.to_string()
                    .contains("a scalar branch the halves may take apart"),
                "{name}: {e}"
            ),
        }
    }
    assert!(
        made_anyway.is_empty(),
        "made with the walk: {made_anyway:?}"
    );
    for (name, words) in [
        (
            "a spilled root constant",
            spilled_root_constant_return_then_count(),
        ),
        ("a constant lane", constant_lane_return_then_count()),
    ] {
        assert!(made(&words, 0, false).is_err(), "{name}: without the walk");
        let l = made(&words, 0, true).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(wave_size(&l.dxil), 32, "{name}");
    }
}
