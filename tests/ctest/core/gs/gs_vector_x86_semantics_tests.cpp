// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// The NEON classes have to give SSE's results where the instructions differ. GSVector4::min/max are
// the exception (see GSVector4_arm64.h).

#include "GS/GSVector.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{
	constexpr s32 kIntMin = static_cast<s32>(0x80000000u);
	const float kNaN = std::numeric_limits<float>::quiet_NaN();

	TEST(GSVectorX86Semantics, TruncatingConversionGivesIntMinOutsideTheInt32RangeAndForNaN)
	{
		const GSVector4i a(GSVector4(1.9f, -1.9f, 3.0e9f, -3.0e9f));
		EXPECT_EQ(a.x, 1);
		EXPECT_EQ(a.y, -1);
		EXPECT_EQ(a.z, kIntMin);
		EXPECT_EQ(a.w, kIntMin);

		const GSVector4i b(GSVector4(kNaN, -2147483648.0f, 2147483520.0f, 2147483648.0f));
		EXPECT_EQ(b.x, kIntMin);
		EXPECT_EQ(b.y, kIntMin); // exactly representable, and x86's indefinite value besides
		EXPECT_EQ(b.z, 2147483520); // the largest float below 2^31
		EXPECT_EQ(b.w, kIntMin);
	}

	TEST(GSVectorX86Semantics, RoundingConversionRoundsToNearestEvenAndGivesIntMinOutsideTheRange)
	{
		const GSVector4i a(GSVector4(1.5f, 2.5f, -1.5f, -2.5f), false);
		EXPECT_EQ(a.x, 2);
		EXPECT_EQ(a.y, 2);
		EXPECT_EQ(a.z, -2);
		EXPECT_EQ(a.w, -2);

		const GSVector4i b(GSVector4(3.0e9f, kNaN, -1.0f, 0.4f), false);
		EXPECT_EQ(b.x, kIntMin);
		EXPECT_EQ(b.y, kIntMin);
		EXPECT_EQ(b.z, -1);
		EXPECT_EQ(b.w, 0);
	}

	// GSRendererSW / GSRendererHWMultiISA turn on the scanline's z overflow path with this test.
	TEST(GSVectorX86Semantics, TheZOverflowTestSeesADepthOfTwoToTheThirtyFirstOrMore)
	{
		EXPECT_EQ(static_cast<u32>(GSVector4i(GSVector4(0.0f, 0.0f, 3.0e9f, 0.0f)).z), 0x80000000u);
		EXPECT_EQ(static_cast<u32>(GSVector4i(GSVector4(0.0f, 0.0f, 4294967040.0f, 0.0f)).z), 0x80000000u);
		EXPECT_NE(static_cast<u32>(GSVector4i(GSVector4(0.0f, 0.0f, 2147483520.0f, 0.0f)).z), 0x80000000u);
	}

	TEST(GSVectorX86Semantics, H2lPutsTheArgumentsHighHalfFirst)
	{
		const GSVector4 r = GSVector4(1.0f, 2.0f, 3.0f, 4.0f).h2l(GSVector4(5.0f, 6.0f, 7.0f, 8.0f));
		EXPECT_EQ(r.x, 7.0f);
		EXPECT_EQ(r.y, 8.0f);
		EXPECT_EQ(r.z, 3.0f);
		EXPECT_EQ(r.w, 4.0f);
	}

#if defined(ARCH_ARM64) // x86 has them only with AVX-512BW (_mm_srlv_epi16)
	TEST(GSVectorX86Semantics, VariableSixteenBitShiftsAreLogicalForSrlAndArithmeticForSra)
	{
		const auto h = [](int v) { return static_cast<short>(static_cast<u16>(v)); };
		const GSVector4i v = GSVector4i::cxpr16(h(0x8000), h(0x8000), h(0x7FFF), h(0xFFFE), h(0x8000), h(0x0100), h(0xFFFF), h(0x1234));
		const GSVector4i n = GSVector4i::cxpr16(1, 15, 3, 1, 0, 8, 4, 4);
		const GSVector4i srl = v.srlv16(n);
		const GSVector4i sra = v.srav16(n);
		const u16 want_srl[8] = {0x4000, 0x0001, 0x0FFF, 0x7FFF, 0x8000, 0x0001, 0x0FFF, 0x0123};
		const u16 want_sra[8] = {0xC000, 0xFFFF, 0x0FFF, 0xFFFF, 0x8000, 0x0001, 0xFFFF, 0x0123};
		for (int i = 0; i < 8; i++)
		{
			EXPECT_EQ(srl.U16[i], want_srl[i]) << "srlv16 lane " << i;
			EXPECT_EQ(sra.U16[i], want_sra[i]) << "srav16 lane " << i;
		}
	}
#endif
} // namespace
