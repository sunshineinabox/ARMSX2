// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "arm64/CodeScheduler-arm64.h"

#include <algorithm>
#include <array>
#include <vector>

namespace a64sched
{
	namespace
	{
		constexpr u64 kFlags = 1ull << 31;
		constexpr u64 kVecMask = ~0ull << 32;

		// Runs are cut at this length, which bounds the quadratic parts of the pass.
		constexpr size_t kMaxRun = 256;

		constexpr u32 Bits(u32 w, int hi, int lo)
		{
			return (w >> lo) & ((1u << (hi - lo + 1)) - 1);
		}

		constexpr s64 SignExtend(u32 v, int bits)
		{
			return static_cast<s64>(static_cast<u64>(v) << (64 - bits)) >> (64 - bits);
		}

		// A general register where encoding 31 is the zero register: reading or writing it is nothing.
		constexpr u64 X(u32 n) { return n == 31 ? 0 : (1ull << n); }
		constexpr u64 V(u32 n) { return 1ull << (32 + n); }

		// `neon` is the NEON/FP issue width (see Insn); by default 1 for anything that touches a vector
		// register, which covers scalar floating point and the lane moves.
		Insn Op(u64 defs, u64 uses, u8 latency, int neon = -1)
		{
			Insn i;
			i.kind = Insn::Kind::Plain;
			i.defs = defs;
			i.uses = uses;
			i.latency = latency;
			i.neon = static_cast<u8>(neon >= 0 ? neon : (((defs | uses) & kVecMask) != 0));
			return i;
		}

		Insn BranchTo(s64 target)
		{
			Insn i;
			i.target = static_cast<s32>(target);
			return i;
		}

		Insn Access(bool load, bool vreg, u32 rt, u32 rn, u32 bytes, s64 offset, bool exact)
		{
			Insn i;
			i.kind = load ? Insn::Kind::Load : Insn::Kind::Store;
			i.base = static_cast<u8>(rn);
			i.bytes = static_cast<u16>(bytes);
			i.offset = static_cast<s32>(offset);
			i.exact = exact;
			i.uses = X(rn);
			if (load)
			{
				i.defs = vreg ? V(rt) : X(rt);
				i.latency = vreg ? 4 : 3;
				if (vreg && bytes >= 16)
				{
					i.occupancy = static_cast<u8>(bytes / 8);
					i.neon = 2;
				}
			}
			else
			{
				i.uses |= vreg ? V(rt) : X(rt);
				i.occupancy = bytes >= 32 ? 2 : 1;
			}
			return i;
		}

		// Size, direction and register file of a single-register load or store (every addressing mode
		// shares the size:V:opc fields). False for prefetches and unallocated encodings.
		bool SingleAccess(u32 w, bool* load, bool* vreg, u32* bytes)
		{
			const u32 size = Bits(w, 31, 30);
			const u32 opc = Bits(w, 23, 22);
			*vreg = (w >> 26) & 1;
			if (!*vreg)
			{
				if ((opc == 2 && size == 3) || (opc == 3 && size >= 2))
					return false;
				*load = opc != 0;
				*bytes = 1u << size;
			}
			else
			{
				if ((opc & 2) && size != 0)
					return false;
				*load = opc & 1;
				*bytes = (opc & 2) ? 16 : (1u << size);
			}
			return true;
		}

		Insn DecodeLoadStore(u32 w)
		{
			const u32 rt = Bits(w, 4, 0);
			const u32 rn = Bits(w, 9, 5);

			// A vixl literal pool starts with `ldr xzr, #size_in_words`; the pool's data follows it.
			if ((w & 0xFF00001F) == 0x5800001F)
			{
				Insn i;
				i.poolWords = Bits(w, 23, 5);
				return i;
			}
			if ((w & 0x3B000000) == 0x18000000) // load literal: PC-relative
				return Insn();

			bool load, vreg;
			u32 bytes;

			// Unsigned scaled offset.
			if ((w & 0x3B000000) == 0x39000000)
			{
				if (rn == 31 || !SingleAccess(w, &load, &vreg, &bytes))
					return Insn();
				return Access(load, vreg, rt, rn, bytes, static_cast<s64>(Bits(w, 21, 10)) * bytes, true);
			}
			// Unscaled offset (LDUR/STUR), pre- and post-index, register offset.
			if ((w & 0x3B200000) == 0x38000000 || (w & 0x3B200C00) == 0x38200800)
			{
				if (rn == 31 || !SingleAccess(w, &load, &vreg, &bytes))
					return Insn();
				const u32 mode = Bits(w, 11, 10);
				if ((w & 0x00200000) != 0) // register offset
				{
					const u32 option = Bits(w, 15, 13);
					if (option != 2 && option != 3 && option != 6 && option != 7)
						return Insn();
					Insn i = Access(load, vreg, rt, rn, bytes, 0, false);
					i.uses |= X(Bits(w, 20, 16));
					return i;
				}
				const s64 imm = SignExtend(Bits(w, 20, 12), 9);
				if (mode == 0)
					return Access(load, vreg, rt, rn, bytes, imm, true);
				if (mode == 2) // unprivileged
					return Insn();
				// Pre-index (3) or post-index (1): the base is written back.
				if (load && !vreg && rt == rn)
					return Insn();
				Insn i = Access(load, vreg, rt, rn, bytes, mode == 3 ? imm : 0, true);
				i.defs |= X(rn);
				return i;
			}
			// Load/store pair: offset, pre/post-index, non-temporal.
			if ((w & 0x3A000000) == 0x28000000)
			{
				const u32 opc = Bits(w, 31, 30);
				const u32 idx = Bits(w, 24, 23);
				const bool l = (w >> 22) & 1;
				const u32 rt2 = Bits(w, 14, 10);
				vreg = (w >> 26) & 1;
				if (rn == 31 || opc == 3)
					return Insn();
				if (!vreg)
				{
					if (opc == 1 && (!l || idx == 0)) // only LDPSW, and it has no non-temporal form
						return Insn();
					bytes = (opc == 2) ? 8 : 4;
				}
				else
				{
					bytes = 4u << opc;
				}
				if (l && rt == rt2)
					return Insn();
				const bool wb = (idx == 1 || idx == 3);
				if (wb && !vreg && (rt == rn || rt2 == rn))
					return Insn();
				const s64 imm = SignExtend(Bits(w, 21, 15), 7) * bytes;
				Insn i = Access(l, vreg, rt, rn, bytes * 2, idx == 1 ? 0 : imm, true);
				if (l)
					i.defs |= vreg ? V(rt2) : X(rt2);
				else
					i.uses |= vreg ? V(rt2) : X(rt2);
				if (wb)
					i.defs |= X(rn);
				return i;
			}
			// AdvSIMD single-structure LD1/ST1 of one lane, no writeback.
			if ((w & 0xBF9F0000) == 0x0D000000)
			{
				if (rn == 31 || (w & 0x00200000) != 0)
					return Insn();
				const u32 opcode = Bits(w, 15, 13);
				const u32 size = Bits(w, 11, 10);
				if (opcode == 0)
					bytes = 1;
				else if (opcode == 2 && (size & 1) == 0)
					bytes = 2;
				else if (opcode == 4 && size == 0)
					bytes = 4;
				else if (opcode == 4 && size == 1 && ((w >> 12) & 1) == 0)
					bytes = 8;
				else
					return Insn();
				load = (w >> 22) & 1;
				Insn i = Access(load, true, rt, rn, bytes, 0, true);
				if (load)
					i.uses |= V(rt); // one lane: the others keep their value
				return i;
			}
			return Insn();
		}

		Insn DecodeDataImm(u32 w)
		{
			const u32 rd = Bits(w, 4, 0);
			const u32 rn = Bits(w, 9, 5);
			const bool sf = w >> 31;

			if ((w & 0x1F800000) == 0x11000000) // add/sub immediate
			{
				const bool s = (w >> 29) & 1;
				if (rn == 31 || (!s && rd == 31))
					return Insn(); // the stack pointer
				return Op(X(rd) | (s ? kFlags : 0), X(rn), 1);
			}
			if ((w & 0x1F800000) == 0x12000000) // logical immediate
			{
				const u32 opc = Bits(w, 30, 29);
				const u32 n = (w >> 22) & 1;
				const u32 imms = Bits(w, 15, 10);
				if (!sf && n)
					return Insn();
				if (opc != 3 && rd == 31)
					return Insn();
				// DecodeBitMasks: the element size comes from N:NOT(imms), and an all-ones element is reserved.
				const u32 lenBits = (n << 6) | (~imms & 0x3F);
				if (lenBits < 2)
					return Insn();
				const u32 levels = (1u << (31 - __builtin_clz(lenBits))) - 1;
				if ((imms & levels) == levels)
					return Insn();
				return Op(X(rd) | (opc == 3 ? kFlags : 0), X(rn), 1);
			}
			if ((w & 0x1F800000) == 0x12800000) // move wide
			{
				const u32 opc = Bits(w, 30, 29);
				if (opc == 1 || (!sf && Bits(w, 22, 21) >= 2))
					return Insn();
				return Op(X(rd), opc == 3 ? X(rd) : 0, 1); // MOVK keeps the other halfwords
			}
			if ((w & 0x1F800000) == 0x13000000) // bitfield
			{
				const u32 opc = Bits(w, 30, 29);
				if (opc == 3 || ((w >> 22) & 1) != static_cast<u32>(sf))
					return Insn();
				if (!sf && ((Bits(w, 21, 16) | Bits(w, 15, 10)) & 0x20))
					return Insn();
				return Op(X(rd), X(rn) | (opc == 1 ? X(rd) : 0), 2); // BFM keeps the other bits
			}
			if ((w & 0x9F000000) == 0x10000000) // ADR
			{
				const s64 bytes = SignExtend((Bits(w, 23, 5) << 2) | Bits(w, 30, 29), 21);
				return (bytes & 3) ? Insn() : BranchTo(bytes / 4);
			}
			return Insn();
		}

		Insn DecodeDataReg(u32 w)
		{
			const u32 rd = Bits(w, 4, 0);
			const u32 rn = Bits(w, 9, 5);
			const u32 rm = Bits(w, 20, 16);
			const bool sf = w >> 31;

			if ((w & 0x1F000000) == 0x0A000000) // logical, shifted register
			{
				const u32 imm6 = Bits(w, 15, 10);
				if (!sf && (imm6 & 32))
					return Insn();
				return Op(X(rd) | (Bits(w, 30, 29) == 3 ? kFlags : 0), X(rn) | X(rm), imm6 ? 2 : 1);
			}
			if ((w & 0x1F200000) == 0x0B000000) // add/sub, shifted register
			{
				const u32 imm6 = Bits(w, 15, 10);
				if (Bits(w, 23, 22) == 3 || (!sf && (imm6 & 32)))
					return Insn();
				return Op(X(rd) | (((w >> 29) & 1) ? kFlags : 0), X(rn) | X(rm), imm6 ? 2 : 1);
			}
			if ((w & 0x1F200000) == 0x0B200000) // add/sub, extended register
			{
				const bool s = (w >> 29) & 1;
				if (rn == 31 || (!s && rd == 31) || Bits(w, 23, 22) != 0 || Bits(w, 12, 10) > 4)
					return Insn();
				return Op(X(rd) | (s ? kFlags : 0), X(rn) | X(rm), 2);
			}
			if ((w & 0x1FE00000) == 0x1A800000) // conditional select
			{
				if (((w >> 29) & 1) || ((w >> 11) & 1))
					return Insn();
				return Op(X(rd), X(rn) | X(rm) | kFlags, 1);
			}
			if ((w & 0x7FE00000) == 0x1B000000) // MADD/MSUB
			{
				// Cortex-A53 erratum 835769: a 64-bit multiply-accumulate right after a load or store can be wrong,
				// so never move one with a real accumulator.
				if (((w >> 31) & 1) && Bits(w, 14, 10) != 31)
					return Insn();
				return Op(X(rd), X(rn) | X(rm) | X(Bits(w, 14, 10)), 3);
			}
			return Insn();
		}

		Insn DecodeSimd(u32 w)
		{
			const u32 rd = Bits(w, 4, 0);
			const u32 rn = Bits(w, 9, 5);
			const u32 rm = Bits(w, 20, 16);
			const bool q = (w >> 30) & 1;
			const bool u = (w >> 29) & 1;
			const u32 size = Bits(w, 23, 22);
			// A 128-bit vector operation issues alone; a 64-bit one pairs. (Lane moves are 64-bit: Op's default.)
			const auto Vec = [q](u64 defs, u64 uses, u8 latency) { return Op(defs, uses, latency, q ? 2 : 1); };

			if ((w & 0x9F200400) == 0x0E200400) // three same
			{
				const u32 opcode = Bits(w, 15, 11);
				if (opcode >= 0x18) // floating point: size<1> selects, size<0> is the precision
				{
					if ((size & 1) && !q)
						return Insn();
					const u32 key = (u << 6) | ((size >> 1) << 5) | opcode;
					switch (key)
					{
						case (0 << 6) | (0 << 5) | 0x18: // FMAXNM
						case (0 << 6) | (1 << 5) | 0x18: // FMINNM
						case (0 << 6) | (0 << 5) | 0x1A: // FADD
						case (0 << 6) | (1 << 5) | 0x1A: // FSUB
						case (0 << 6) | (0 << 5) | 0x1B: // FMULX
						case (0 << 6) | (0 << 5) | 0x1E: // FMAX
						case (0 << 6) | (1 << 5) | 0x1E: // FMIN
						case (1 << 6) | (0 << 5) | 0x1B: // FMUL
						case (1 << 6) | (0 << 5) | 0x1A: // FADDP
							return Vec(V(rd), V(rn) | V(rm), 4);
						case (0 << 6) | (0 << 5) | 0x1C: // FCMEQ
						case (1 << 6) | (0 << 5) | 0x1C: // FCMGE
						case (1 << 6) | (1 << 5) | 0x1C: // FCMGT
							return Vec(V(rd), V(rn) | V(rm), 2);
						case (0 << 6) | (0 << 5) | 0x19: // FMLA
						case (0 << 6) | (1 << 5) | 0x19: // FMLS
							return Vec(V(rd), V(rd) | V(rn) | V(rm), 4);
						default:
							return Insn();
					}
				}
				switch (opcode)
				{
					case 0x10: // ADD/SUB
					case 0x06: // CMGT/CMHI
					case 0x07: // CMGE/CMHS
					case 0x11: // CMTST/CMEQ
						if (size == 3 && !q)
							return Insn();
						return Vec(V(rd), V(rn) | V(rm), 2);
					case 0x0C: // SMAX/UMAX
					case 0x0D: // SMIN/UMIN
						if (size == 3)
							return Insn();
						return Vec(V(rd), V(rn) | V(rm), 2);
					case 0x03: // logical; BSL/BIT/BIF also read the destination, and take a cycle more
						return (u && size != 0) ? Vec(V(rd), V(rd) | V(rn) | V(rm), 2) : Vec(V(rd), V(rn) | V(rm), 1);
					default:
						return Insn();
				}
			}
			if ((w & 0x9F3E0C00) == 0x0E200800) // two-register miscellaneous
			{
				const u32 opcode = Bits(w, 16, 12);
				const u32 key = (u << 6) | ((size >> 1) << 5) | opcode;
				if (opcode >= 0x0C) // floating point
				{
					if ((size & 1) && !q)
						return Insn();
					switch (key)
					{
						case (0 << 6) | (1 << 5) | 0x0C: // FCMGT #0
						case (1 << 6) | (1 << 5) | 0x0C: // FCMGE #0
						case (0 << 6) | (1 << 5) | 0x0D: // FCMEQ #0
						case (1 << 6) | (1 << 5) | 0x0D: // FCMLE #0
						case (0 << 6) | (1 << 5) | 0x0E: // FCMLT #0
							return Vec(V(rd), V(rn), 2);
						case (0 << 6) | (1 << 5) | 0x0F: // FABS
						case (1 << 6) | (1 << 5) | 0x0F: // FNEG
							return Vec(V(rd), V(rn), 2);
						case (0 << 6) | (0 << 5) | 0x1D: // SCVTF
						case (1 << 6) | (0 << 5) | 0x1D: // UCVTF
						case (0 << 6) | (1 << 5) | 0x1B: // FCVTZS
						case (1 << 6) | (1 << 5) | 0x1B: // FCVTZU
						case (0 << 6) | (1 << 5) | 0x1D: // FRECPE
						case (1 << 6) | (1 << 5) | 0x1D: // FRSQRTE
							return Vec(V(rd), V(rn), 4);
						default:
							return Insn();
					}
				}
				if (size == 3 && !q)
					return Insn();
				switch ((u << 5) | opcode)
				{
					case (0 << 5) | 0x08: // CMGT #0
					case (1 << 5) | 0x08: // CMGE #0
					case (0 << 5) | 0x09: // CMEQ #0
					case (1 << 5) | 0x09: // CMLE #0
					case (0 << 5) | 0x0A: // CMLT #0
					case (0 << 5) | 0x0B: // ABS
					case (1 << 5) | 0x0B: // NEG
						return Vec(V(rd), V(rn), 2);
					case (1 << 5) | 0x05: // NOT (size 0; RBIT is size 1)
						return size == 0 ? Vec(V(rd), V(rn), 1) : Insn();
					default:
						return Insn();
				}
			}
			if ((w & 0x9F3E0C00) == 0x0E300800) // across lanes
			{
				const u32 opcode = Bits(w, 16, 12);
				if (size == 3 || (size == 2 && !q))
					return Insn();
				switch ((u << 5) | opcode)
				{
					case (0 << 5) | 0x1B: // ADDV
					case (0 << 5) | 0x0A: // SMAXV
					case (1 << 5) | 0x0A: // UMAXV
					case (0 << 5) | 0x1A: // SMINV
					case (1 << 5) | 0x1A: // UMINV
					case (0 << 5) | 0x03: // SADDLV
					case (1 << 5) | 0x03: // UADDLV
						return Vec(V(rd), V(rn), 3);
					default:
						return Insn();
				}
			}
			if ((w & 0x9FE08400) == 0x0E000400) // copy: DUP, INS, UMOV, SMOV
			{
				const u32 imm5 = Bits(w, 20, 16);
				const u32 imm4 = Bits(w, 14, 11);
				if ((imm5 & 0xF) == 0)
					return Insn();
				const u32 esize = __builtin_ctz(imm5); // 0 byte, 1 half, 2 word, 3 doubleword
				if (u) // INS (element)
					return q ? Op(V(rd), V(rd) | V(rn), 2) : Insn();
				switch (imm4)
				{
					case 0x0: // DUP (element)
					case 0x1: // DUP (general)
						if (esize == 3 && !q)
							return Insn();
						return imm4 ? Vec(V(rd), X(rn), 3) : Vec(V(rd), V(rn), 2);
					case 0x3: // INS (general)
						return q ? Op(V(rd), V(rd) | X(rn), 2) : Insn();
					case 0x5: // SMOV: to a W register from a byte or half, to an X register also from a word
						return (esize < (q ? 3u : 2u)) ? Op(X(rd), V(rn), 2) : Insn();
					case 0x7: // UMOV: to a W register up to a word, to an X register only a doubleword
						return (q ? esize == 3 : esize < 3) ? Op(X(rd), V(rn), 2) : Insn();
					default:
						return Insn();
				}
			}
			if ((w & 0xBF208C00) == 0x0E000800) // ZIP/UZP/TRN
			{
				const u32 opcode = Bits(w, 14, 12);
				if (opcode == 0 || opcode == 4 || (size == 3 && !q))
					return Insn();
				return Vec(V(rd), V(rn) | V(rm), 2);
			}
			if ((w & 0xBFE08400) == 0x2E000000) // EXT
			{
				if (!q && ((w >> 14) & 1))
					return Insn();
				return Vec(V(rd), V(rn) | V(rm), 2);
			}
			if ((w & 0x9F800400) == 0x0F000400) // shift by immediate, or modified immediate when immh is 0
			{
				const u32 immh = Bits(w, 22, 19);
				if (immh == 0)
				{
					const u32 cmode = Bits(w, 15, 12);
					if ((w >> 11) & 1)
						return Insn();
					if (u && cmode == 0xF && !q)
						return Insn();
					const bool orrBic = (cmode & 0x9) == 0x1 || (cmode & 0xD) == 0x9;
					return Vec(V(rd), orrBic ? V(rd) : 0, 2);
				}
				const u32 opcode = Bits(w, 15, 11);
				switch ((u << 5) | opcode)
				{
					case (0 << 5) | 0x00: // SSHR
					case (1 << 5) | 0x00: // USHR
					case (0 << 5) | 0x0A: // SHL
						return (immh & 8) && !q ? Insn() : Vec(V(rd), V(rn), 2);
					case (1 << 5) | 0x0A: // SLI
					case (1 << 5) | 0x08: // SRI
						return (immh & 8) && !q ? Insn() : Vec(V(rd), V(rd) | V(rn), 1);
					case (0 << 5) | 0x14: // SSHLL
					case (1 << 5) | 0x14: // USHLL
						return (immh & 8) ? Insn() : Op(V(rd), V(rn), 2, 2); // the long result is 128 bits
					default:
						return Insn();
				}
			}
			// Vector, and scalar, by element: FMUL/FMULX/FMLA/FMLS. Rm is the full M:Rm field for these.
			const bool byElemVec = (w & 0x9F000400) == 0x0F000000;
			const bool byElemScalar = (w & 0xDF000400) == 0x5F000000;
			if (byElemVec || byElemScalar)
			{
				const u32 opcode = Bits(w, 15, 12);
				if (size < 2 || (size == 3 && ((w >> 21) & 1)) || (byElemVec && size == 3 && !q))
					return Insn();
				switch ((u << 4) | opcode)
				{
					case (0 << 4) | 0x9: // FMUL
					case (1 << 4) | 0x9: // FMULX
						return Op(V(rd), V(rn) | V(rm), 4, byElemVec && q ? 2 : 1);
					case (0 << 4) | 0x1: // FMLA
					case (0 << 4) | 0x5: // FMLS
						return Op(V(rd), V(rd) | V(rn) | V(rm), 4, byElemVec && q ? 2 : 1);
					default:
						return Insn();
				}
			}
			return Insn();
		}

		Insn DecodeFp(u32 w)
		{
			const u32 rd = Bits(w, 4, 0);
			const u32 rn = Bits(w, 9, 5);
			const u32 rm = Bits(w, 20, 16);
			const u32 ftype = Bits(w, 23, 22);

			switch (w & 0xFFFFFC00)
			{
				case 0x1E260000: // FMOV Wd, Sn
				case 0x9E660000: // FMOV Xd, Dn
				case 0x1E270000: // FMOV Sd, Wn
				case 0x9E670000: // FMOV Dd, Xn
					return (w & 0x10000) ? Op(V(rd), X(rn), 2) : Op(X(rd), V(rn), 2);
				default:
					break;
			}
			if (ftype > 1)
				return Insn();
			if ((w & 0xFF200C00) == 0x1E200800) // two source
			{
				switch (Bits(w, 15, 12))
				{
					case 0x1: // FDIV
					{
						Insn i = Op(V(rd), V(rn) | V(rm), 10);
						i.divider = true;
						return i;
					}
					case 0x0: case 0x2: case 0x3: case 0x4: case 0x5: case 0x6: case 0x7: case 0x8:
						return Op(V(rd), V(rn) | V(rm), 4);
					default:
						return Insn();
				}
			}
			if ((w & 0xFF207C00) == 0x1E204000) // one source
			{
				switch (Bits(w, 20, 15))
				{
					case 0x0: // FMOV
					case 0x1: // FABS
					case 0x2: // FNEG
						return Op(V(rd), V(rn), 2);
					case 0x3: // FSQRT
					{
						Insn i = Op(V(rd), V(rn), 12);
						i.divider = true;
						return i;
					}
					default:
						return Insn();
				}
			}
			if ((w & 0xFF20FC07) == 0x1E202000) // FCMP/FCMPE, with a register or #0.0
				return Op(kFlags, V(rn) | (((w >> 3) & 1) ? 0 : V(rm)), 3);
			if ((w & 0xFF201FE0) == 0x1E201000) // FMOV (scalar, immediate)
				return Op(V(rd), 0, 2);
			return Insn();
		}

		struct Scratch
		{
			std::vector<Insn> insns;
			std::vector<u8> boundary;
		};

		int RegionOf(const Insn& i, const MemoryModel& m)
		{
			for (size_t k = 0; k < m.count; k++)
			{
				const BaseRegion& b = m.bases[k];
				if (b.reg != i.base)
					continue;
				if (i.exact)
					return i.offset >= b.split ? b.regionAbove : b.region;
				return b.region == b.regionAbove ? b.region : -1;
			}
			return -1;
		}

		bool MayOverlap(const Insn& a, int ra, const Insn& b, int rb)
		{
			if (ra < 0 || rb < 0)
				return true;
			if (ra != rb)
				return false;
			if (a.base != b.base || !a.exact || !b.exact)
				return true;
			return a.offset < b.offset + static_cast<s64>(b.bytes) && b.offset < a.offset + static_cast<s64>(a.bytes);
		}

		// In-order dual-issue model: 128-bit NEON ops take both pipes, one load and one store a cycle, and
		// FDIV/FSQRT wait on the divider.
		struct Issue
		{
			std::array<u32, 64> ready; // valid where `written` has the bit: 0 otherwise
			u64 written = 0;
			u32 cycle = 0;
			u32 slots = 0;
			u32 neonUsed = 0;
			bool loadUsed = false;
			bool storeUsed = false;
			u32 loadFree = 0;
			u32 storeFree = 0;
			u32 neonFree = 0;
			u32 dividerFree = 0;

			// When the registers `i` reads are ready. In a list schedule this is fixed once all of `i`'s
			// predecessors are placed, as whatever writes a register it reads is one of them.
			u32 OperandsReady(const Insn& i) const
			{
				u32 t = 0;
				for (u64 m = i.uses & written; m; m &= m - 1)
					t = std::max(t, ready[__builtin_ctzll(m)]);
				return t;
			}

			u32 WhenIssued(const Insn& i) const { return WhenIssued(i, OperandsReady(i)); }

			u32 WhenIssued(const Insn& i, u32 operands) const
			{
				const bool load = i.kind == Insn::Kind::Load;
				const bool store = i.kind == Insn::Kind::Store;
				u32 t = std::max(cycle, operands);
				if (t == cycle && (slots >= 2 || neonUsed + i.neon > 2 || (load && loadUsed) || (store && storeUsed)))
					t = cycle + 1;
				if (load)
					t = std::max(t, loadFree);
				if (store)
					t = std::max(t, storeFree);
				if (i.neon)
					t = std::max(t, neonFree);
				if (i.divider)
					t = std::max(t, dividerFree);
				return t;
			}

			void Place(const Insn& i) { Place(i, WhenIssued(i)); }

			// `t` is WhenIssued(i).
			void Place(const Insn& i, u32 t)
			{
				if (t != cycle)
				{
					cycle = t;
					slots = 0;
					neonUsed = 0;
					loadUsed = storeUsed = false;
				}
				slots++;
				neonUsed += i.neon;
				if (i.kind == Insn::Kind::Load)
				{
					loadUsed = true;
					loadFree = t + i.occupancy;
					if (i.neon)
						neonFree = t + i.occupancy;
				}
				else if (i.kind == Insn::Kind::Store)
				{
					storeUsed = true;
					storeFree = t + i.occupancy;
				}
				if (i.divider)
					dividerFree = t + i.latency;
				for (u64 m = i.defs; m; m &= m - 1)
					ready[__builtin_ctzll(m)] = t + i.latency;
				written |= i.defs;
			}
		};

		// A set of run positions: bit k is instruction k of the run.
		template <size_t Words>
		struct RunSet
		{
			std::array<u64, Words> w;
			void Clear() { w.fill(0); }
			void Set(size_t k) { w[k >> 6] |= 1ull << (k & 63); }
			void Merge(const RunSet& o)
			{
				for (size_t i = 0; i < w.size(); i++)
					w[i] |= o.w[i];
			}
			u32 Count() const
			{
				u32 c = 0;
				for (u64 x : w)
					c += static_cast<u32>(__builtin_popcountll(x));
				return c;
			}
			template <typename F>
			void ForEach(F&& f) const
			{
				for (size_t i = 0; i < w.size(); i++)
					for (u64 x = w[i]; x; x &= x - 1)
						f(i * 64 + static_cast<size_t>(__builtin_ctzll(x)));
			}
		};

		// Per-thread scratch for one run of up to 64 * Words instructions: no allocation while scheduling.
		template <size_t Words>
		struct RunScratch
		{
			static constexpr size_t kMax = Words * 64;
			std::array<RunSet<Words>, kMax> preds; // instructions that must come first
			std::array<RunSet<Words>, kMax> succs;
			std::array<RunSet<Words>, 64> readers; // per register: who read it since it was last written
			std::array<s16, 64> lastWriter;
			std::array<u16, kMax> mem; // the run's loads and stores, in order
			std::array<int, kMax> region;
			std::array<u32, kMax> height;
			std::array<u16, kMax> waiting;
			std::array<u16, kMax> avail;
			std::array<u32, kMax> operands; // OperandsReady, once the instruction is available
			std::array<u32, kMax> order;
			std::array<u32, kMax> identity;
			std::array<u32, kMax> words;
		};

		// Reorders insns[0, n) (code words alongside); returns true when it changed the order. Most runs
		// are short, and take the one-word sets.
		template <size_t Words>
		bool ScheduleRun(u32* code, const Insn* insns, size_t n, const MemoryModel& memory, Stats* stats)
		{
			thread_local RunScratch<Words> scratch;
			RunScratch<Words>& s = scratch; // one thread-local lookup, not one per access

			u64 touched = 0;
			for (size_t k = 0; k < n; k++)
			{
				touched |= insns[k].uses | insns[k].defs;
				s.preds[k].Clear();
				s.succs[k].Clear();
			}
			for (u64 m = touched; m; m &= m - 1)
			{
				const int r = __builtin_ctzll(m);
				s.readers[r].Clear();
				s.lastWriter[r] = -1;
			}

			auto edge = [&s](size_t from, size_t to) {
				s.preds[to].Set(from);
				s.succs[from].Set(to);
			};

			size_t memCount = 0;
			for (size_t k = 0; k < n; k++)
			{
				const Insn& i = insns[k];
				for (u64 m = i.uses; m; m &= m - 1)
				{
					const int r = __builtin_ctzll(m);
					if (s.lastWriter[r] >= 0)
						edge(static_cast<size_t>(s.lastWriter[r]), k);
				}
				for (u64 m = i.defs; m; m &= m - 1)
				{
					const int r = __builtin_ctzll(m);
					if (s.lastWriter[r] >= 0)
						edge(static_cast<size_t>(s.lastWriter[r]), k);
					s.readers[r].ForEach([&](size_t j) { edge(j, k); });
				}
				for (u64 m = i.uses; m; m &= m - 1)
					s.readers[__builtin_ctzll(m)].Set(k);
				for (u64 m = i.defs; m; m &= m - 1)
				{
					const int r = __builtin_ctzll(m);
					s.lastWriter[r] = static_cast<s16>(k);
					s.readers[r].Clear();
				}

				const bool load = i.kind == Insn::Kind::Load;
				const bool store = i.kind == Insn::Kind::Store;
				if (load || store)
				{
					s.region[k] = RegionOf(i, memory);
					for (size_t mi = 0; mi < memCount; mi++)
					{
						const size_t j = s.mem[mi];
						const Insn& p = insns[j];
						if (!(p.kind == Insn::Kind::Store || store))
							continue;
						if (MayOverlap(p, s.region[j], i, s.region[k]))
							edge(j, k);
					}
					s.mem[memCount++] = static_cast<u16>(k);
				}
			}

			// Longest latency path to the end of the run, the list scheduler's priority.
			for (size_t k = n; k-- > 0;)
			{
				u32 h = 0;
				s.succs[k].ForEach([&](size_t j) { h = std::max(h, s.height[j]); });
				s.height[k] = h + insns[k].latency;
			}

			// The ready list, kept in priority order (longest path first, then the original order), so the
			// first entry that can issue this cycle is the pick, and a stall is the only full scan.
			Issue issue;
			size_t availCount = 0;
			const auto ready = [&](size_t k, u32 operands) {
				s.operands[k] = operands;
				size_t a = availCount++;
				for (; a > 0; a--)
				{
					const u16 p = s.avail[a - 1];
					if (s.height[p] > s.height[k] || (s.height[p] == s.height[k] && p < k))
						break;
					s.avail[a] = p;
				}
				s.avail[a] = static_cast<u16>(k);
			};
			for (size_t k = 0; k < n; k++)
			{
				s.waiting[k] = static_cast<u16>(s.preds[k].Count());
				if (s.waiting[k] == 0)
					ready(k, 0);
			}

			for (size_t placed = 0; placed < n; placed++)
			{
				size_t best = 0;
				u32 bestT = UINT32_MAX;
				for (size_t a = 0; a < availCount; a++)
				{
					const u16 k = s.avail[a];
					const u32 t = issue.WhenIssued(insns[k], s.operands[k]);
					if (t < bestT)
					{
						best = a;
						bestT = t;
						if (t == issue.cycle)
							break; // nothing issues earlier, and the rest rank lower
					}
				}
				const u16 k = s.avail[best];
				std::copy(s.avail.begin() + best + 1, s.avail.begin() + availCount, s.avail.begin() + best);
				availCount--;
				s.order[placed] = k;
				issue.Place(insns[k], bestT);
				s.succs[k].ForEach([&](size_t j) {
					if (--s.waiting[j] == 0)
						ready(j, issue.OperandsReady(insns[j]));
				});
			}

			for (size_t k = 0; k < n; k++)
				s.identity[k] = static_cast<u32>(k);
			const u32 before = ModelCycles(insns, s.identity.data(), n);
			const u32 after = issue.cycle + 1;

			if (stats)
			{
				stats->runs++;
				stats->cyclesBefore += before;
				stats->cyclesAfter += std::min(before, after);
			}
			if (after >= before)
				return false;

			for (size_t k = 0; k < n; k++)
				s.words[k] = code[s.order[k]];
			std::copy(s.words.begin(), s.words.begin() + n, code);
			return true;
		}
	} // namespace

	Insn Decode(u32 w)
	{
		const u32 op0 = Bits(w, 28, 25);

		// Branches, exception generation, system: never moved. Record direct targets.
		if ((w & 0x7C000000) == 0x14000000) // B, BL
			return BranchTo(SignExtend(Bits(w, 25, 0), 26));
		if ((w & 0xFF000010) == 0x54000000) // B.cond
			return BranchTo(SignExtend(Bits(w, 23, 5), 19));
		if ((w & 0x7E000000) == 0x34000000) // CBZ, CBNZ
			return BranchTo(SignExtend(Bits(w, 23, 5), 19));
		if ((w & 0x7E000000) == 0x36000000) // TBZ, TBNZ
			return BranchTo(SignExtend(Bits(w, 18, 5), 14));

		if ((op0 & 0x5) == 0x4) // x1x0: loads and stores
			return DecodeLoadStore(w);
		if ((op0 & 0xE) == 0x8) // 100x: data processing, immediate
			return DecodeDataImm(w);
		if ((op0 & 0x7) == 0x5) // x101: data processing, register
			return DecodeDataReg(w);
		if ((op0 & 0x7) == 0x7) // x111: SIMD and floating point
		{
			if ((w & 0x5F000000) == 0x1E000000) // scalar floating point (not AdvSIMD)
				return DecodeFp(w);
			return DecodeSimd(w);
		}
		return Insn();
	}

	u32 ModelCycles(const Insn* insns, const u32* order, size_t count)
	{
		Issue issue;
		for (size_t k = 0; k < count; k++)
			issue.Place(insns[order[k]]);
		return count ? issue.cycle + 1 : 0;
	}

	void Schedule(u32* code, size_t words, const size_t* entries, size_t entryCount, const MemoryModel& memory,
		Stats* stats, bool keepMoveWide, const void* runsAt)
	{
		thread_local Scratch scratch;
		Scratch& s = scratch;
		if (s.insns.size() < words)
			s.insns.resize(words); // grown, never shrunk: every entry used is decoded below
		s.boundary.assign(words + 1, 0);
		if (stats)
			stats->words += words;

		const uptr lo = reinterpret_cast<uptr>(runsAt ? runsAt : code);
		const uptr hi = lo + words * sizeof(u32);
		for (size_t k = 0; k < entryCount; k++)
			if (entries[k] < words)
				s.boundary[entries[k]] = 1;

		for (size_t k = 0; k < words; k++)
		{
			s.insns[k] = (keepMoveWide && (code[k] & 0x1F800000) == 0x12800000) ? Insn() : Decode(code[k]);
			const Insn& i = s.insns[k];
			if (i.target != Insn::kNoTarget)
			{
				const s64 t = static_cast<s64>(k) + i.target;
				if (t >= 0 && static_cast<size_t>(t) < words)
					s.boundary[t] = 1;
			}
			if (i.poolWords)
			{
				// Skip the pool's data, and treat any code address it holds as an entry point. The marker's count
				// includes the marker itself, so the data is the poolWords - 1 words after it.
				const size_t end = std::min(words, k + std::max<size_t>(i.poolWords, 1));
				for (size_t d = k + 1; d < end; d++)
				{
					s.insns[d] = Insn();
					if (d + 1 < end)
					{
						const uptr v = static_cast<uptr>(code[d]) | (static_cast<uptr>(code[d + 1]) << 32);
						if (v >= lo && v < hi && ((v - lo) & 3) == 0)
							s.boundary[(v - lo) / sizeof(u32)] = 1;
					}
				}
				k = end - 1;
			}
		}

		size_t start = 0;
		auto flush = [&](size_t end) {
			while (end - start >= 2)
			{
				const size_t n = std::min(end - start, kMaxRun);
				const bool changed = n <= 64 ? ScheduleRun<1>(code + start, s.insns.data() + start, n, memory, stats) :
				                               ScheduleRun<kMaxRun / 64>(code + start, s.insns.data() + start, n, memory, stats);
				if (changed && stats)
					stats->reordered++;
				start += n;
			}
			start = end;
		};
		for (size_t k = 0; k < words; k++)
		{
			if (s.boundary[k])
				flush(k);
			if (s.insns[k].kind == Insn::Kind::Barrier)
			{
				flush(k);
				start = k + 1;
			}
		}
		flush(words);
	}
} // namespace a64sched
