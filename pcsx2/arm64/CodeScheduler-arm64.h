// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

#include "common/Pcsx2Defs.h"

#include <cstddef>

// Post-pass that reorders emitted A64 code for in-order cores, so independent work fills the stalls
// between dependent steps. It only permutes straight-line runs and keeps all dependencies.
namespace a64sched
{
	/// What a pinned base register points at, so accesses through two of them can be told apart. Offsets
	/// below `split` are in `region`, offsets at or past it in `regionAbove`.
	struct BaseRegion
	{
		u8 reg;
		u8 region;
		u8 regionAbove;
		s64 split;
	};

	/// Accesses through a base not listed here may touch anything.
	struct MemoryModel
	{
		const BaseRegion* bases = nullptr;
		size_t count = 0;
	};

	/// One decoded instruction: what it reads and writes, and how long its result takes.
	struct Insn
	{
		enum class Kind : u8
		{
			Barrier, // ends a run and stays where it is
			Plain,
			Load,
			Store,
		};

		static constexpr s32 kNoTarget = INT32_MIN;

		u64 defs = 0; // bit n: Xn (0-30), bit 31: NZCV, bit 32+n: Vn
		u64 uses = 0;
		s32 offset = 0;
		s32 target = kNoTarget; // branch or ADR target, in words from this instruction
		u32 poolWords = 0; // a vixl literal-pool marker: this many data words follow
		u16 bytes = 0; // bytes the access covers from `offset`
		Kind kind = Kind::Barrier;
		u8 latency = 1;
		u8 base = 0; // load/store base register
		u8 neon = 0; // NEON/FP issue width: 1 for a 64-bit operation, 2 for a 128-bit one, which issues alone
		u8 occupancy = 1; // cycles a load or store holds its pipe; a 128-bit load holds the NEON pipes as well
		bool exact = false; // the access is base + offset, offset known
		bool divider = false; // FDIV/FSQRT: the divider is not pipelined
	};

	Insn Decode(u32 word);

	struct Stats
	{
		u64 words = 0;
		u64 runs = 0; // runs of two or more instructions
		u64 reordered = 0; // runs whose order changed
		u64 cyclesBefore = 0; // A55 model, all runs
		u64 cyclesAfter = 0;
	};

	/// Reorders the runs in code[0, words). `entries` are word indices entered from outside.
	/// `keepMoveWide` pins MOVZ/MOVN/MOVK.
	void Schedule(u32* code, size_t words, const size_t* entries, size_t entryCount, const MemoryModel& memory,
		Stats* stats = nullptr, bool keepMoveWide = false);

	/// Timing-model cycles for insns[order[0..count)] from an idle pipeline.
	u32 ModelCycles(const Insn* insns, const u32* order, size_t count);
} // namespace a64sched
