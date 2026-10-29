// SPDX-FileCopyrightText: 2026 ARMSX2 Contributors
// SPDX-License-Identifier: GPL-3.0+

// A zero clear over a whole cached target is done on the GPU. One with no target, or covering only
// part of one, still goes to local memory.

#include "gs_hw_draw_harness.h"

#include "GS/Renderers/HW/GSTextureCache.h"

#include <cstring>

using namespace GSHWDrawHarness;

namespace
{
	constexpr int kFrameW = 640;
	constexpr int kFrameH = 448;
	constexpr u32 kFramePages = 10 * 14; // FBW 10, 448 rows of 32-row pages
	constexpr u32 kZPage = 0x100; // Environment()'s ZBUF.ZBP
	constexpr u32 kZBlock = kZPage * 32;

	class GSZeroClear : public Fixture
	{
	protected:
		/// A full-frame sprite of one colour. With `z` set, depth is written too, at that value.
		void DrawFrame(u32 rgba, bool write_z = false, u32 z = 0)
		{
			Packet p;
			Environment(p, 0, PSMCT32);

			GIFReg r = {};
			r.RGBAQ.U32[0] = rgba;
			r.RGBAQ.Q = 1.0f;
			p.Reg(GIF_A_D_REG_RGBAQ, r);

			if (write_z)
			{
				r.U64 = 0;
				r.ZBUF.ZBP = kZPage;
				r.ZBUF.PSM = PSMZ32;
				r.ZBUF.ZMSK = 0;
				p.Reg(GIF_A_D_REG_ZBUF_1, r);
			}

			p.Vertex(0, 0, z, 0, 0);
			p.Vertex(kFrameW << 4, kFrameH << 4, z, 0, 0);
			GIFRegPRIM prim = {};
			prim.PRIM = GS_SPRITE;
			p.Send(*m_gs, prim);
		}

		static GSTextureCache::Target* FrameTarget()
		{
			return g_texture_cache->GetExactTarget(0, 10, GSTextureCache::RenderTarget, 0);
		}

		static GSTextureCache::Target* DepthTarget()
		{
			return g_texture_cache->GetExactTarget(kZBlock, 10, GSTextureCache::DepthStencil, kZBlock);
		}

		void Fill(u32 first_page, u8 value)
		{
			std::memset(m_gs->m_mem.m_vm8 + static_cast<size_t>(first_page) * GS_PAGE_SIZE, value,
				static_cast<size_t>(kFramePages) * GS_PAGE_SIZE);
			m_gs->m_mem.MarkPageRangeWritten(first_page, kFramePages);
		}

		bool Holds(u32 first_page, u8 value) const
		{
			const u8* p = m_gs->m_mem.m_vm8 + static_cast<size_t>(first_page) * GS_PAGE_SIZE;
			for (size_t i = 0; i < static_cast<size_t>(kFramePages) * GS_PAGE_SIZE; i++)
			{
				if (p[i] != value)
					return false;
			}
			return true;
		}
	};
} // namespace

TEST_F(GSZeroClear, OverAWholeColourTargetClearsItOnTheGpu)
{
	BringUp();
	DrawFrame(0x80204060);
	GSTextureCache::Target* const before = FrameTarget();
	ASSERT_NE(before, nullptr);

	Fill(0, 0xAB);
	DrawFrame(0);

	GSTextureCache::Target* const after = FrameTarget();
	ASSERT_EQ(after, before) << "the target was retired";
	EXPECT_EQ(after->m_texture->GetState(), GSTexture::State::Cleared);
	EXPECT_EQ(after->m_texture->GetClearColor(), 0u);
	EXPECT_TRUE(Holds(0, 0xAB)) << "local memory under the target was written";
}

TEST_F(GSZeroClear, OverWholeColourAndDepthTargetsClearsBothOnTheGpu)
{
	BringUp();
	DrawFrame(0x80204060, true, 0x1234);
	GSTextureCache::Target* const rt = FrameTarget();
	GSTextureCache::Target* const ds = DepthTarget();
	ASSERT_NE(rt, nullptr);
	ASSERT_NE(ds, nullptr);

	Fill(0, 0xAB);
	Fill(kZPage, 0xCD);
	DrawFrame(0x80204060, true, 0);

	ASSERT_EQ(FrameTarget(), rt) << "the colour target was retired";
	ASSERT_EQ(DepthTarget(), ds) << "the depth target was retired";
	EXPECT_EQ(ds->m_texture->GetState(), GSTexture::State::Cleared);
	EXPECT_TRUE(Holds(0, 0xAB)) << "local memory under the colour target was written";
	EXPECT_TRUE(Holds(kZPage, 0xCD)) << "local memory under the depth target was written";
}

TEST_F(GSZeroClear, WithNoTargetStillGoesToLocalMemory)
{
	BringUp();
	Fill(0, 0xAB);
	DrawFrame(0);

	EXPECT_EQ(FrameTarget(), nullptr) << "a target was made for a clear nothing used yet";
	EXPECT_TRUE(Holds(0, 0x00)) << "the clear did not reach local memory";
}
