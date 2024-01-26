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

struct ConsoleSize
{
	int rows;
	int columns;
};

enum class StandardStream
{
	Out,
	Err
};

/**
 * @brief Returns the current size of the terminal window (in characters)
 * @return Console size
 */
ConsoleSize getConsoleSize(StandardStream ss);

/**
 * @brief Determines whether @c stdout is attached to a terminal or redirected
 * @return @c true if stdout is attached to a terminal, @c false otherwise
 */
bool isStdOutAttachedToTerminal();

/**
 * @brief Determines whether @c stderr is attached to a terminal or redirected
 * @return @c true if stderr is attached to a terminal, @c false otherwise
 */
bool isStdErrAttachedToTerminal();

#ifdef _WIN32
// Windows (x64 and x86)

#define NOMINMAX
#define _WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <io.h>
#include <stdio.h>

ConsoleSize getConsoleSize(StandardStream ss)
{
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	ConsoleSize cs;

	if (ss == StandardStream::Out)
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	else if (ss == StandardStream::Err)
		GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &csbi);

	cs.columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
	cs.rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
	return cs;
}

bool isStdOutAttachedToTerminal()
{
	DWORD st;
	const int fn = _fileno(stdout);
	HANDLE hd = reinterpret_cast<HANDLE>(_get_osfhandle(fn));
	return _isatty(fn) && (hd != INVALID_HANDLE_VALUE) && GetConsoleMode(hd, &st);
}

bool isStdErrAttachedToTerminal()
{
	DWORD st;
	const int fn = _fileno(stderr);
	HANDLE hd = reinterpret_cast<HANDLE>(_get_osfhandle(fn));
	return _isatty(fn) && (hd != INVALID_HANDLE_VALUE) && GetConsoleMode(hd, &st);
}

#elif __unix__ || __linux__ || __APPLE__
// Linux, BSD, Mac OSX

#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h>

ConsoleSize getConsoleSize(StandardStream ss)
{
	winsize w;
	if (ss == StandardStream::Out)
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
	else if (ss == StandardStream::Err)
		ioctl(STDERR_FILENO, TIOCGWINSZ, &w);

	return ConsoleSize{w.ws_row, w.ws_col};
}

bool isStdOutAttachedToTerminal()
{
	return isatty(fileno(stdout));
}

bool isStdErrAttachedToTerminal()
{
	return isatty(fileno(stderr));
}

#endif

#include "ProgressBar.hpp"
#include "common/CompilerSpecific.hpp"

#include <algorithm>
#include <cstdio>

namespace cadet
{

ProgressBar::ProgressBar() : _os(nullptr), _progress(0.0), _minUpdate(0.005), _minTime(0.1), _buffer(0), _barRatio(0.6)
{
	useStdOut();
}

ProgressBar::~ProgressBar() CADET_NOEXCEPT
{
}

ProgressBar& ProgressBar::minProgressUpdate(double minUpdate)
{
	_minUpdate = minUpdate;
	return *this;
}

ProgressBar& ProgressBar::minTimeUpdate(double minTime)
{
	_minTime = minTime;
	return *this;
}

ProgressBar& ProgressBar::useStdOut()
{
	if (isStdOutAttachedToTerminal())
	{
		_os = &std::cout;
		const ConsoleSize cs = getConsoleSize(StandardStream::Out);
		setup(cs.columns);
	}
	else
		_os = nullptr;

	return *this;
}

ProgressBar& ProgressBar::useStdErr()
{
	if (isStdErrAttachedToTerminal())
	{
		_os = &std::cerr;
		const ConsoleSize cs = getConsoleSize(StandardStream::Err);
		setup(cs.columns);
	}
	else
		_os = nullptr;

	return *this;
}

ProgressBar& ProgressBar::disable()
{
	_os = nullptr;
	return *this;
}

ProgressBar& ProgressBar::barWidthRatio(double width)
{
	_barRatio = width;
	return *this;
}

void ProgressBar::setup(int columns)
{
	// Apply default terminal size
	if (columns == 0)
		columns = 80;

	_termWidth = columns;

	// Add one element for terminal '\0' char, and one for carriage return '\r'
	_buffer.resize(columns + 2);

	// Add prefix to buffer
	std::snprintf(_buffer.data(), _buffer.size(), "\r [");

	// Add terminal '\0' char
	_buffer.back() = '\0';
}

void ProgressBar::begin()
{
	_startTime = std::chrono::steady_clock::now();
}

void ProgressBar::print() const
{
	print(std::chrono::steady_clock::now(), true);
}

void ProgressBar::print(typename std::chrono::steady_clock::time_point curTime, bool arrowTip) const
{
	if (cadet_unlikely(!_os))
		return;

	cadet_assert((_progress >= 0.0) && (_progress <= 1.0));

	// Extrapolate remaining time
	const double elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(curTime - _startTime).count();
	const double remainingSec = (std::max)(0.0, elapsedSec * (1.0 / (std::max)(_progress, 1e-10) - 1.0));

	// Compute size for the bar, reserve 4 chars for braces and spaces
	const int barSize = _barRatio * (_termWidth - 4);

	// Skip carriage return and prefix
	int pos = 3;

	// Add bar
	const int barProgWidth = (std::max)(0.0, _progress * barSize - 1);
	char* const ptrBar = _buffer.data() + pos;
	for (int i = 0; i < barProgWidth; ++i)
		ptrBar[i] = '=';

	// Add the tip
	if (arrowTip)
		ptrBar[barProgWidth] = '>';
	else
		ptrBar[barProgWidth] = '=';

	// Fill remaining bar with space
	for (int i = barProgWidth + 1; i < barSize; ++i)
		ptrBar[i] = ' ';

	pos += barSize;
	const int textSize = std::snprintf(_buffer.data() + pos, _buffer.size() - pos, "] %4.1f%% (%5.1f>%5.1f s) %s",
									   _progress * 100.0, elapsedSec, remainingSec, _message.c_str());

	if (textSize + pos < static_cast<int>(_buffer.size()) - 1)
	{
		// Fill remaining buffer
		char* const ptrBuffer = _buffer.data();
		for (int i = textSize + pos; i < static_cast<int>(_buffer.size()) - 1; ++i)
			ptrBuffer[i] = ' ';
	}

	(*_os) << _buffer.data() << std::flush;
}

void ProgressBar::update(char const* message)
{
	update(_progress, message);
}

void ProgressBar::update(double progress, char const* message)
{
	const double deltaProgress = progress - _progress;
	const typename std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
	const double deltaTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - _lastUpdateTime).count();

	// Update if enough progress has been made or enough time has elapsed
	if ((deltaTime >= _minTime) || (deltaProgress >= _minUpdate))
	{
		_message = message;
		_progress = progress;

		print(curTime, true);
		_lastUpdateTime = curTime;
	}
}

void ProgressBar::finish()
{
	_progress = 1.0;
	print(std::chrono::steady_clock::now(), false);

	if (_os)
		(*_os) << std::endl;
}

void ProgressBar::finish(char const* message)
{
	_message = message;
	finish();
}

} // namespace cadet
