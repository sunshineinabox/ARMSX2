// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// vu-aot defaults on (off under 2 GB of RAM) and turns on the VU program cache; the cache's own
// setting stays off.

#include <gtest/gtest.h>

#include "Config.h"

#include "common/HostSys.h"
#include "common/MemorySettingsInterface.h"
#include "common/SettingsWrapper.h"

namespace
{
	Pcsx2Config::RecompilerOptions Load(MemorySettingsInterface& si)
	{
		Pcsx2Config::RecompilerOptions rc;
		SettingsLoadWrapper wrap(si);
		rc.LoadSave(wrap);
		return rc;
	}
} // namespace

TEST(VuAotSetting, DefaultsOffOnlyBelowTwoGigabytesOfRam)
{
	using RC = Pcsx2Config::RecompilerOptions;
	constexpr u64 GB = 1ull << 30;
	EXPECT_FALSE(RC::DefaultVUAot(GB)) << "the RG DS: 975 MB";
	EXPECT_FALSE(RC::DefaultVUAot(2 * GB - 1));
	EXPECT_TRUE(RC::DefaultVUAot(2 * GB));
	EXPECT_TRUE(RC::DefaultVUAot(4 * GB)) << "the MANGMI Air X: 3.6 GB";
	EXPECT_TRUE(RC::DefaultVUAot(16 * GB));
	EXPECT_TRUE(RC::DefaultVUAot(0)) << "RAM the host could not report keeps the old default";
}

TEST(VuAotSetting, DefaultsByThisHostsRamAndTheProgramCacheSettingStaysOff)
{
	const Pcsx2Config::RecompilerOptions rc;
	EXPECT_EQ(rc.EnableVUAot, Pcsx2Config::RecompilerOptions::DefaultVUAot(GetPhysicalMemory()));
	EXPECT_FALSE(rc.EnableVUProgramCache);
	EXPECT_EQ(rc.UseVUProgramCache(), rc.EnableVUAot) << "vu-aot is bound to the cache";
}

TEST(VuAotSetting, ASettingsFileWithoutTheKeyGetsTheDefault)
{
	MemorySettingsInterface si;
	EXPECT_EQ(Load(si).EnableVUAot, Pcsx2Config::RecompilerOptions().EnableVUAot);
	EXPECT_FALSE(Load(si).EnableVUProgramCache);
}

TEST(VuAotSetting, ReadsAnExplicitValueEitherWay)
{
	for (const bool on : {false, true})
	{
		MemorySettingsInterface si;
		si.SetBoolValue("EmuCore/CPU/Recompiler", "EnableVUAot", on);
		const Pcsx2Config::RecompilerOptions rc = Load(si);
		EXPECT_EQ(rc.EnableVUAot, on);
		EXPECT_EQ(rc.UseVUProgramCache(), on) << "off with the program cache setting off: no cache";
	}
}

TEST(VuAotSetting, OffWithTheProgramCacheSettingOnKeepsTheCache)
{
	MemorySettingsInterface si;
	si.SetBoolValue("EmuCore/CPU/Recompiler", "EnableVUAot", false);
	si.SetBoolValue("EmuCore/CPU/Recompiler", "EnableVUProgramCache", true);
	const Pcsx2Config::RecompilerOptions rc = Load(si);
	EXPECT_FALSE(rc.EnableVUAot);
	EXPECT_TRUE(rc.UseVUProgramCache());
}

TEST(VuAotSetting, IsWrittenUnderItsKey)
{
	for (const bool on : {false, true})
	{
		Pcsx2Config::RecompilerOptions rc;
		rc.EnableVUAot = on;
		MemorySettingsInterface si;
		SettingsSaveWrapper wrap(si);
		rc.LoadSave(wrap);

		bool saved = !on;
		ASSERT_TRUE(si.GetBoolValue("EmuCore/CPU/Recompiler", "EnableVUAot", &saved));
		EXPECT_EQ(saved, on);
	}
}

TEST(VuAotSetting, ChangingItIsAChangeOfTheCpuOptions)
{
	// (VMManager::CheckForCPUConfigChanges resets the VU recompilers when the options differ)
	Pcsx2Config::RecompilerOptions a, b;
	EXPECT_TRUE(a == b);
	b.EnableVUAot = !a.EnableVUAot;
	EXPECT_TRUE(a != b);
}
