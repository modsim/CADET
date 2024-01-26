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

/**
 * @file
 * Progress bar for cadet-cli.
 */

#ifndef CADETCLI_PROGRESSBAR_HPP_
#define CADETCLI_PROGRESSBAR_HPP_

#include "cadet/cadetCompilerInfo.hpp"

#include <iostream>
#include <chrono>
#include <vector>

namespace cadet
{

class ProgressBar
{
public:
	ProgressBar();
	~ProgressBar() CADET_NOEXCEPT;

	/**
	 * @brief Signals the beginning of the process
	 */
	void begin();

	/**
	 * @brief Forces a printing of the progress bar
	 */
	void print() const;

	/**
	 * @brief Updates the progress and prints it
	 * @details Only prints the progress bar if enough progress has been made
	 *          or a certain amount of time has been elapsed.
	 * @param[in] progress Current progress in percent (between @c 0.0 and @c 1.0)
	 * @param[in] message Message to put behind the bar
	 */
	void update(double progress, char const* message);

	/**
	 * @brief Updates the progress and prints it
	 * @details Only prints the progress bar if enough progress has been made
	 *          or a certain amount of time has been elapsed.
	 * @param[in] message Message to put behind the bar
	 */
	void update(char const* message);

	/**
	 * @brief Signals the end of the process
	 * @details Finishes the progress bar and moves to a new line
	 */
	void finish();

	/**
	 * @brief Signals the end of the process
	 * @details Finishes the progress bar and moves to a new line
	 * @param[in] message Message to put behind the bar
	 */
	void finish(char const* message);

	/**
	 * @brief Sets the minimum amount of progress that allows to reprint the bar
	 * @param[in] minUpdate Minimum amount of update in percent (between @c 0.0 and @c 1.0)
	 * @return Current ProgressBar object to allow for chaining
	 */
	ProgressBar& minProgressUpdate(double minUpdate);

	/**
	 * @brief Sets the minimum amount of elapsed time that allows to reprint the bar
	 * @param[in] minTime Minimum amount of time in seconds
	 * @return Current ProgressBar object to allow for chaining
	 */
	ProgressBar& minTimeUpdate(double minTime);

	/**
	 * @brief Sets @c cout as the output stream
	 * @return Current ProgressBar object to allow for chaining
	 */
	ProgressBar& useStdOut();

	/**
	 * @brief Sets @c cerr as the output stream
	 * @return Current ProgressBar object to allow for chaining
	 */
	ProgressBar& useStdErr();

	/**
	 * @brief Disables the ProgressBar
	 * @return Current ProgressBar object to allow for chaining
	 */
	ProgressBar& disable();

	/**
	 * @brief Sets the width of the actual progress bar in the terminal
	 * @param[in] width Width in percent (between @c 0.0 and @c 1.0)
	 * @return Current ProgressBar object to allow for chaining
	 */
	ProgressBar& barWidthRatio(double width);

protected:
	std::ostream* _os; //!< Stream to print to
	double _progress;  //!< Current progress in percent (between 0.0 and 1.0)
	double _minUpdate; //!< Minimum amount of progress update to trigger printing
	double _minTime;   //!< Minimum amount of elapsed time between printing
	typename std::chrono::steady_clock::time_point _startTime;      //!< Start time of process
	typename std::chrono::steady_clock::time_point _lastUpdateTime; //!< Time of last update
	int _termWidth;                                                 //!< Number of columns of terminal
	mutable std::vector<char> _buffer;                              //!< Buffer for printing the progress bar
	std::string _message;                                           //!< Message to display behind the bar
	double _barRatio; //!< Percentage of terminal width used for the bar (between 0.0 and 1.0)

	void setup(int columns);
	void print(typename std::chrono::steady_clock::time_point curTime, bool arrowTip) const;
};

} // namespace cadet

#endif // CADETCLI_PROGRESSBAR_HPP_
