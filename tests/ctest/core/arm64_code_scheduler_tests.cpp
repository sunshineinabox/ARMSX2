// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// a64sched: random straight-line code must leave the same registers, flags and memory after scheduling.

#include "common/Pcsx2Defs.h"

#if defined(__aarch64__)

#include "arm64/CodeScheduler-arm64.h"

#include "Arm64JitBuffer.h"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <vector>

#ifdef __APPLE__
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif

namespace
{
	using a64sched::Insn;

	constexpr u64 Xr(u32 n) { return 1ull << n; }
	constexpr u64 Vr(u32 n) { return 1ull << (32 + n); }
	constexpr u64 kNZCV = 1ull << 31;

	// The bases the generated loads and stores use, and what they point at in the run below: x19's buffer
	// holds x29's from byte 2048 on, as VURegs is followed by VU1's data memory.
	constexpr u32 kSplit = 2048;
	const a64sched::BaseRegion kBases[] = {
		{19, 0, 1, kSplit},
		{29, 1, 1, INT64_MAX},
		{24, 2, 2, INT64_MAX},
		{25, 3, 3, INT64_MAX},
	};
	const a64sched::MemoryModel kModel{kBases, std::size(kBases)};

	TEST(CodeScheduler, DecodesWhatTheRecompilerEmits)
	{
		// fminnm v20.4s, v0.4s, v25.4s
		Insn i = a64sched::Decode(0x4eb9c414);
		EXPECT_EQ(i.kind, Insn::Kind::Plain);
		EXPECT_EQ(i.defs, Vr(20));
		EXPECT_EQ(i.uses, Vr(0) | Vr(25));
		EXPECT_EQ(i.latency, 4);

		// mov v3.d[0], v9.d[0]: an INS writes one lane, so it also reads its destination
		i = a64sched::Decode(0x6e080523);
		EXPECT_EQ(i.defs, Vr(3));
		EXPECT_EQ(i.uses, Vr(3) | Vr(9));

		// ldr q4, [x19, #0x50]
		i = a64sched::Decode(0x3dc01664);
		EXPECT_EQ(i.kind, Insn::Kind::Load);
		EXPECT_EQ(i.base, 19);
		EXPECT_TRUE(i.exact);
		EXPECT_EQ(i.offset, 0x50);
		EXPECT_EQ(i.bytes, 16u);
		EXPECT_EQ(i.defs, Vr(4));

		// ldr q9, [x29, x9]: the offset is a register
		i = a64sched::Decode(0x3ce96ba9);
		EXPECT_EQ(i.kind, Insn::Kind::Load);
		EXPECT_FALSE(i.exact);
		EXPECT_EQ(i.uses, Xr(29) | Xr(9));

		// tst w14, #0xffff writes only the flags
		i = a64sched::Decode(0x72003ddf);
		EXPECT_EQ(i.defs, kNZCV);
		EXPECT_EQ(i.uses, Xr(14));

		// b.pl, a literal-pool marker, an ADRP and an add to sp are never moved
		EXPECT_EQ(a64sched::Decode(0x54000045).kind, Insn::Kind::Barrier);
		EXPECT_EQ(a64sched::Decode(0x54000045).target, 2);
		EXPECT_EQ(a64sched::Decode(0x5800013f).poolWords, 9u);
		EXPECT_EQ(a64sched::Decode(0x90000000).kind, Insn::Kind::Barrier);
		EXPECT_EQ(a64sched::Decode(0x910043ff).kind, Insn::Kind::Barrier);

		// Cortex-A53 erratum 835769: a 64-bit multiply-accumulate stays where it is; a 64-bit MUL (accumulator
		// XZR) and a 32-bit MADD still move.
		EXPECT_EQ(a64sched::Decode(0x9B0A2D28).kind, Insn::Kind::Barrier); // madd x8, x9, x10, x11
		EXPECT_NE(a64sched::Decode(0x9B0A7D28).kind, Insn::Kind::Barrier); // mul x8, x9, x10
		EXPECT_NE(a64sched::Decode(0x1B0A2D28).kind, Insn::Kind::Barrier); // madd w8, w9, w10, w11
	}

	// --- Encoders for the generated sequences and the harness around them. ---

	u32 LdStXImm(bool load, u32 rt, u32 rn, u32 off) { return 0xF9000000u | (load << 22) | ((off / 8) << 10) | (rn << 5) | rt; }
	u32 LdStQImm(bool load, u32 rt, u32 rn, u32 off) { return 0x3D800000u | (load << 22) | ((off / 16) << 10) | (rn << 5) | rt; }
	u32 StpPre(bool vreg, u32 rt, u32 rt2) { return (vreg ? 0x6D800000u : 0xA9800000u) | (0x7Eu << 15) | (rt2 << 10) | (31 << 5) | rt; }
	u32 LdpPost(bool vreg, u32 rt, u32 rt2) { return (vreg ? 0x6CC00000u : 0xA8C00000u) | (0x02u << 15) | (rt2 << 10) | (31 << 5) | rt; }

	struct State
	{
		u64 x[16]; // x1-x15 (x[0] unused: x0 holds this struct's address)
		alignas(16) u8 v[32][16];
		u64 nzcv;
		u64 bases[4]; // x19, x24, x25, x29
	};

	class Generator
	{
	public:
		explicit Generator(u32 seed) : m_rng(seed) {}

		std::vector<u32> Sequence(size_t length)
		{
			std::vector<u32> out;
			while (out.size() < length)
			{
				const u32 roll = Pick(100);
				if (roll < 2 && out.size() + 6 < length)
				{
					// A vixl literal pool behind a branch over it: the marker holds the pool's size in words, itself
					// included (vixl's LiteralPool::Emit).
					const u32 words = 1 + Pick(4);
					out.push_back(0x14000000u | (words + 2));
					out.push_back(0x5800001Fu | ((words + 1) << 5));
					for (u32 k = 0; k < words; k++)
						out.push_back(static_cast<u32>(m_rng()));
					continue;
				}
				if (roll < 4)
				{
					out.push_back(0x14000001u); // b to the next instruction: a run boundary
					continue;
				}
				out.push_back(Op());
			}
			return out;
		}

	private:
		u32 Pick(u32 n) { return static_cast<u32>(m_rng() % n); }
		u32 G() { return 1 + Pick(15); } // x1-x15
		u32 Vn() { return Pick(32); }

		u32 Mem()
		{
			static constexpr u32 bases[] = {19, 24, 25, 29};
			const u32 rn = bases[Pick(4)];
			const u32 load = Pick(2);
			switch (Pick(6))
			{
				case 0: // ldr/str q, unsigned offset
					return 0x3D800000u | (load << 22) | (Pick(256) << 10) | (rn << 5) | Vn();
				case 1: // ldr/str s
					return 0xBD000000u | (load << 22) | (Pick(1024) << 10) | (rn << 5) | Vn();
				case 2: // ldr/str w
					return 0xB9000000u | (load << 22) | (Pick(1024) << 10) | (rn << 5) | G();
				case 3: // ldrh/strh
					return 0x79000000u | (load << 22) | (Pick(2048) << 10) | (rn << 5) | G();
				case 4: // ldp/stp q, signed offset
				{
					const u32 a = Vn();
					const u32 b = (a + 1 + Pick(31)) & 31;
					return 0xAD000000u | (load << 22) | (Pick(64) << 15) | (b << 10) | (rn << 5) | a;
				}
				default: // ld1/st1 {v.s}[lane], [base]
					return 0x0D008000u | (Pick(2) << 30) | (Pick(2) << 12) | (load << 22) | (rn << 5) | Vn();
			}
		}

		u32 Op()
		{
			const u32 d = Vn(), n = Vn(), m = Vn();
			switch (Pick(30))
			{
				case 0: return 0x11000000u | (Pick(2) << 30) | (Pick(4096) << 10) | (G() << 5) | G(); // add/sub w imm
				case 1: return 0x31000000u | (Pick(2) << 30) | (Pick(4096) << 10) | (G() << 5) | G(); // adds/subs w imm
				case 2: // and/orr/eor/ands w, #bitmask (32-bit element, s+1 ones rotated by r)
					return 0x12000000u | (Pick(4) << 29) | (Pick(32) << 16) | (Pick(31) << 10) | (G() << 5) | G();
				case 3: return 0x52800000u | (Pick(2) << 29) | (Pick(2) << 21) | ((m_rng() & 0xFFFF) << 5) | G(); // movz/movk w
				case 4: return 0x13000000u | (Pick(3) << 29) | (Pick(32) << 16) | (Pick(32) << 10) | (G() << 5) | G(); // sbfm/bfm/ubfm w
				case 5: return 0x0A000000u | (Pick(4) << 29) | (Pick(2) << 21) | (G() << 16) | (Pick(32) << 10) | (G() << 5) | G(); // logical w, shifted
				case 6: return 0x0B000000u | (Pick(4) << 29) | (G() << 16) | (Pick(32) << 10) | (G() << 5) | G(); // add/sub(s) w, shifted
				case 7: return 0x1A800000u | (Pick(2) << 30) | (G() << 16) | (Pick(16) << 12) | (Pick(2) << 10) | (G() << 5) | G(); // csel/csinc
				case 8: return 0x1B000000u | (G() << 16) | (G() << 10) | (G() << 5) | G(); // madd w
				case 9: // FP three same .4s
				{
					static constexpr u32 ops[] = {0x4E20D400, 0x4EA0D400, 0x6E20DC00, 0x4E20C400, 0x4EA0C400, 0x4E20F400, 0x4EA0F400,
						0x4E20DC00, 0x6E20D400, 0x4E20E400, 0x6E20E400, 0x6EA0E400, 0x4E20CC00, 0x4EA0CC00};
					return ops[Pick(std::size(ops))] | (m << 16) | (n << 5) | d;
				}
				case 10: // integer three same .4s and logical .16b
				{
					static constexpr u32 ops[] = {0x4EA08400, 0x6EA08400, 0x4EA03400, 0x6EA08C00, 0x4EA06C00, 0x6EA06C00, 0x4EA06400,
						0x6EA06400, 0x4E201C00, 0x4E601C00, 0x4EA01C00, 0x6E201C00, 0x6E601C00, 0x6EA01C00, 0x6EE01C00};
					return ops[Pick(std::size(ops))] | (m << 16) | (n << 5) | d;
				}
				case 11: // two-register misc .4s
				{
					static constexpr u32 ops[] = {0x4EA0A800, 0x4EA09800, 0x4EA08800, 0x4EA0D800, 0x4EA0E800, 0x4EA0F800, 0x6EA0F800,
						0x4E21D800, 0x4EA1B800, 0x4EA0B800, 0x6EA0B800, 0x6E205800};
					return ops[Pick(std::size(ops))] | (n << 5) | d;
				}
				case 12: // fmul/fmla/fmls .4s by element
				{
					static constexpr u32 ops[] = {0x4F809000, 0x4F801000, 0x4F805000};
					return ops[Pick(3)] | (Pick(2) << 21) | (Pick(2) << 11) | (m << 16) | (n << 5) | d;
				}
				case 13: return 0x4E040400u | (Pick(4) << 19) | (n << 5) | d; // dup .4s, v.s[i]
				case 14: return 0x4E040C00u | (G() << 5) | d; // dup .4s, w
				case 15: return 0x6E040400u | (Pick(4) << 19) | (Pick(4) << 13) | (n << 5) | d; // ins s[i], s[j]
				case 16: return 0x4E041C00u | (Pick(4) << 19) | (G() << 5) | d; // ins s[i], w
				case 17: return 0x0E043C00u | (Pick(4) << 19) | (n << 5) | G(); // umov w, s[i]
				case 18: return 0x6E000000u | (m << 16) | (Pick(4) << 13) | (n << 5) | d; // ext .16b, #4k
				case 19: return 0x4E800800u | (Pick(2) ? 0x3000u : 0x1000u) | (m << 16) | (n << 5) | d; // zip1/uzp1 .4s
				case 20: // sshr/ushr/shl/sli .4s
				{
					const u32 sh = 1 + Pick(31);
					switch (Pick(4))
					{
						case 0: return 0x4F000400u | ((64 - sh) << 16) | (n << 5) | d;
						case 1: return 0x6F000400u | ((64 - sh) << 16) | (n << 5) | d;
						case 2: return 0x4F005400u | ((32 + sh) << 16) | (n << 5) | d;
						default: return 0x6F005400u | ((32 + sh) << 16) | (n << 5) | d;
					}
				}
				case 21: return 0x4EB1B800u | (n << 5) | d; // addv s, .4s
				case 22: // movi .2d #0, or orr .4s #imm8 (reads its destination)
					return Pick(2) ? (0x6F00E400u | d) : (0x4F001400u | (Pick(8) << 16) | (Pick(32) << 5) | d);
				case 23: // scalar FP two source: fmul fadd fsub fminnm fmaxnm
				{
					static constexpr u32 ops[] = {0x1E200800, 0x1E202800, 0x1E203800, 0x1E207800, 0x1E206800};
					return ops[Pick(5)] | (m << 16) | (n << 5) | d;
				}
				case 24: return Pick(2) ? (0x1E202000u | (m << 16) | (n << 5)) : (0x1E202008u | (n << 5)); // fcmp s
				case 25: return Pick(2) ? (0x1E260000u | (n << 5) | G()) : (0x1E270000u | (G() << 5) | d); // fmov w<->s
				default:
					return Mem();
			}
		}

		std::mt19937 m_rng;
	};

	// Runs `seq` on `state` through a harness that loads x1-x15, v0-v31, NZCV and the four bases from it,
	// and stores the registers and flags back.
	void Execute(JitBuffer& buf, const std::vector<u32>& seq, State& state)
	{
		std::vector<u32> code;
		for (u32 r = 29; r >= 19; r -= 2)
			code.push_back(StpPre(false, r, r + 1));
		for (u32 r = 14; r >= 8; r -= 2)
			code.push_back(StpPre(true, r, r + 1));
		code.push_back(LdStXImm(true, 1, 0, offsetof(State, nzcv)));
		code.push_back(0xD51B4201u); // msr nzcv, x1
		for (u32 r = 0; r < 32; r++)
			code.push_back(LdStQImm(true, r, 0, offsetof(State, v) + r * 16));
		static constexpr u32 baseRegs[] = {19, 24, 25, 29};
		for (u32 k = 0; k < 4; k++)
			code.push_back(LdStXImm(true, baseRegs[k], 0, offsetof(State, bases) + k * 8));
		for (u32 r = 1; r < 16; r++)
			code.push_back(LdStXImm(true, r, 0, offsetof(State, x) + r * 8));
		code.insert(code.end(), seq.begin(), seq.end());
		for (u32 r = 1; r < 16; r++)
			code.push_back(LdStXImm(false, r, 0, offsetof(State, x) + r * 8));
		for (u32 r = 0; r < 32; r++)
			code.push_back(LdStQImm(false, r, 0, offsetof(State, v) + r * 16));
		code.push_back(0xD53B4201u); // mrs x1, nzcv
		code.push_back(LdStXImm(false, 1, 0, offsetof(State, nzcv)));
		for (u32 r = 8; r <= 14; r += 2)
			code.push_back(LdpPost(true, r, r + 1));
		for (u32 r = 19; r <= 29; r += 2)
			code.push_back(LdpPost(false, r, r + 1));
		code.push_back(0xD65F03C0u); // ret

		ASSERT_LE(code.size() * 4, buf.size());
#ifdef __APPLE__
		pthread_jit_write_protect_np(0);
#endif
		std::memcpy(buf.ptr(), code.data(), code.size() * 4);
#ifdef __APPLE__
		pthread_jit_write_protect_np(1);
		sys_icache_invalidate(buf.ptr(), code.size() * 4);
#else
		__builtin___clear_cache(static_cast<char*>(buf.ptr()), static_cast<char*>(buf.ptr()) + code.size() * 4);
#endif
		reinterpret_cast<void (*)(State*)>(buf.ptr())(&state);
	}

	TEST(CodeScheduler, ScheduledCodeComputesTheSame)
	{
		JitBuffer buf(1 << 16);
		ASSERT_NE(buf.ptr(), nullptr);

		// x19's buffer is 8 KB and x29 points 2 KB into it (the split above); x24 and x25 have their own.
		std::vector<u8> mem(8192 + 2 * 4096);
		u8* const regs = mem.data();
		u8* const mvu = regs + 8192;
		u8* const glob = mvu + 4096;

		a64sched::Stats stats;
		u32 checked = 0;
		for (u32 seed = 1; seed <= 3000; seed++)
		{
			Generator gen(seed);
			const std::vector<u32> original = gen.Sequence(2 + seed % 160);
			std::vector<u32> scheduled = original;
			const bool keepMoveWide = (seed & 1) == 0;
			a64sched::Schedule(scheduled.data(), scheduled.size(), nullptr, 0, kModel, &stats, keepMoveWide);
			for (size_t k = 0; keepMoveWide && k < original.size(); k++)
			{
				if ((original[k] & 0x1F800000) == 0x12800000)
					ASSERT_EQ(scheduled[k], original[k]) << "a move-wide moved under keepMoveWide, seed " << seed;
			}

			std::mt19937 rng(seed * 7919u);
			State init;
			std::memset(&init, 0, sizeof(init));
			for (auto& x : init.x)
				x = (static_cast<u64>(rng()) << 32) | rng();
			for (auto& v : init.v)
				for (auto& b : v)
					b = static_cast<u8>(rng());
			init.nzcv = static_cast<u64>(rng() & 0xF) << 28;
			init.bases[0] = reinterpret_cast<u64>(regs);
			init.bases[1] = reinterpret_cast<u64>(mvu);
			init.bases[2] = reinterpret_cast<u64>(glob);
			init.bases[3] = reinterpret_cast<u64>(regs + kSplit);
			std::vector<u8> memInit(mem.size());
			for (auto& b : memInit)
				b = static_cast<u8>(rng());

			State a = init, b = init;
			std::memcpy(mem.data(), memInit.data(), mem.size());
			Execute(buf, original, a);
			const std::vector<u8> memA = mem;
			std::memcpy(mem.data(), memInit.data(), mem.size());
			Execute(buf, scheduled, b);

			ASSERT_EQ(std::memcmp(a.x, b.x, sizeof(a.x)), 0) << "general registers differ, seed " << seed;
			ASSERT_EQ(std::memcmp(a.v, b.v, sizeof(a.v)), 0) << "vector registers differ, seed " << seed;
			ASSERT_EQ(a.nzcv, b.nzcv) << "flags differ, seed " << seed;
			ASSERT_EQ(memA, mem) << "memory differs, seed " << seed;
			checked++;
		}
		EXPECT_EQ(checked, 3000u);
		// The comparison means something only if the pass moved code.
		EXPECT_GT(stats.reordered, 1000u);
		EXPECT_LT(stats.cyclesAfter, stats.cyclesBefore);
	}
} // namespace

#endif
