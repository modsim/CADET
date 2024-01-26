// =============================================================================
//  CADET
//
//  Copyright © 2008-2022: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#ifdef _WIN32
// Windows (x64 and x86)

#define NOMINMAX
#define _WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <stdio.h>

namespace
{
volatile bool stopExecution = false;

BOOL WINAPI signalHandler(DWORD fdwCtrlType)
{
	switch (fdwCtrlType)
	{
	case CTRL_C_EVENT:
		stopExecution = true;
		return TRUE;

	case CTRL_CLOSE_EVENT:
		stopExecution = true;
		return TRUE;
	}

	// Pass other signals to the next handler
	return FALSE;
}
} // namespace

namespace cadet
{
bool installSignalHandler()
{
	if (SetConsoleCtrlHandler(signalHandler, TRUE))
		return true;

	return false;
}

bool stopExecutionRequested()
{
	return stopExecution;
}

} // namespace cadet

#elif __unix__ || __linux__ || __APPLE__
// Linux, BSD, Mac OSX

#include <signal.h>
#include <stdio.h>

namespace
{
volatile sig_atomic_t stopExecution = 0;

void signalHandler(int signal)
{
	if (signal == SIGINT)
		stopExecution = 1;
}
} // namespace

namespace cadet
{
bool installSignalHandler()
{
	struct sigaction sigbreak;
	sigbreak.sa_handler = &signalHandler;
	sigemptyset(&sigbreak.sa_mask);
	sigbreak.sa_flags = 0;

	if (sigaction(SIGINT, &sigbreak, NULL) != 0)
		return false;

	return true;
}

bool stopExecutionRequested()
{
	return stopExecution == 1;
}

} // namespace cadet

#endif
