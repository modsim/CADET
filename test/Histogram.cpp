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

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>

namespace cadet
{
namespace test
{

void printHistogram(const std::vector<double>& data, int nBins, unsigned int skip, unsigned int lane)
{
	double minVal = std::numeric_limits<double>::infinity();
	double maxVal = -std::numeric_limits<double>::infinity();
	unsigned int nValid = 0;

	// Find min and max
	for (std::size_t i = 0; i < data.size() / skip; ++i)
	{
		const double v = std::log10(data[i * skip + lane]);
		if (std::isfinite(v))
		{
			++nValid;
			minVal = std::min(minVal, v);
			maxVal = std::max(maxVal, v);
		}
	}

	// Do binning
	const double binWidth = (maxVal - minVal) / static_cast<double>(nBins);
	std::vector<unsigned int> bins(nBins, 0);
	for (std::size_t i = 0; i < data.size() / skip; ++i)
	{
		const double v = std::log10(data[i * skip + lane]);
		if (!std::isfinite(v))
			continue;

		if (v == maxVal)
			++bins.back();
		else
			++bins[static_cast<int>((v - minVal) / binWidth)];
	}

	const unsigned int consoleWidth = 72;
	const unsigned int width = consoleWidth - (6 + 3 + 6 + 2);

	// Plot
	for (int i = 0; i < nBins; ++i)
	{
		const double low = minVal + binWidth * i;
		const double high = minVal + binWidth * (i + 1);
		std::cout << std::setw(6) << std::setprecision(2) << std::fixed << low << " = " << std::setw(6)
				  << std::setprecision(2) << std::fixed << high;
		std::cout << " |";

		const int len = static_cast<int>(std::ceil(static_cast<double>(width * bins[i]) / static_cast<double>(nValid)));

		for (int j = 0; j < len; ++j)
			std::cout << "*";
		std::cout << "\n";
	}
	std::cout << " N = " << (data.size() / skip) << " (" << nValid
			  << "), * = " << std::ceil(static_cast<double>(nValid) / static_cast<double>(width)) << std::endl;
}

template <typename T> inline double Lerp(T v0, T v1, T t)
{
	return (1 - t) * v0 + t * v1;
}

template <typename T>
std::vector<T> quantile(std::vector<T>& data, const std::vector<T>& probs, unsigned int skip, unsigned int lane)
{
	std::vector<T> d(data.size() / skip);
	for (std::size_t i = 0; i < d.size(); ++i)
		d[i] = data[i * skip + lane];

	std::sort(d.begin(), d.end());
	std::vector<T> quantiles;
	quantiles.reserve(probs.size());

	for (std::size_t i = 0; i < probs.size(); ++i)
	{
		const T poi = Lerp<T>(-0.5, d.size() - 0.5, probs[i]);

		std::size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
		std::size_t right = std::min(int64_t(std::ceil(poi)), int64_t(d.size() - 1));

		const T datLeft = d.at(left);
		const T datRight = d.at(right);

		const T quantile = Lerp<T>(datLeft, datRight, poi - left);

		quantiles.push_back(quantile);
	}

	return quantiles;
}

void printQuantiles(std::vector<double>& data, unsigned int skip, unsigned int lane)
{
	const std::vector<double> probs = {0.5, 0.66, 0.75, 0.9, 0.925, 0.95, 0.975};
	const std::vector<double> quants = quantile(data, probs, skip, lane);

	for (std::size_t i = 0; i < probs.size(); ++i)
	{
		std::cout << " " << std::setw(4) << std::setprecision(1) << std::fixed << probs[i] * 100.0 << " => "
				  << std::setw(10) << std::setprecision(3) << std::scientific << quants[i] << "\n";
	}
	std::cout << std::endl;
}
} // namespace test
} // namespace cadet
