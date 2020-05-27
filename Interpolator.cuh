#ifndef CUCBET_INTERPOLATOR_CUH
#define CUCBET_INTERPOLATOR_CUH

/*
 * This code is borrowed *without* explicit permission from
 * https://bulldozer00.blog/2016/05/10/linear-interpolation-in-c/
 */

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

class Interpolator {
public:
	//On construction, we take in a vector of data point pairs
	//that represent the line we will use to interpolate
	explicit Interpolator(std::vector<std::pair<float, float>> points) : _points(std::move(points)) {
		//Defensive programming. Assume the caller has not sorted the table in
		//in ascending order
		std::sort(_points.begin(), _points.end());

		//Ensure that no 2 adjacent x values are equal,
		//lest we try to divide by zero when we interpolate.
		const float EPSILON{1.0E-8};
		for(std::size_t i=1; i<_points.size(); ++i) {
			float deltaX{std::abs(_points[i].first - _points[i-1].first)};
			if(deltaX < EPSILON ) {
				std::string err{"Potential Divide By Zero: Points " +
				                std::to_string(i-1) + " And " +
				                std::to_string(i) + " Are Too Close In Value"};
				throw std::range_error(err);
			}
		}
	}

	//Computes the corresponding Y value
	//for X using linear interpolation
	float findValue(float x) const {
		//Define a lambda that returns true if the x value
		//of a point pair is < the caller's x value
		auto lessThan =
				[](const std::pair<float, float>& point, float x)
				{return point.first < x;};

		//Find the first table entry whose value is >= caller's x value
		auto iter =
				std::lower_bound(_points.cbegin(), _points.cend(), x, lessThan);

		//If the caller's X value is greater than the largest
		//X value in the table, we can't interpolate.
		if(iter == _points.cend()) {
			return (_points.cend() - 1)->second;
		}

		//If the caller's X value is less than the smallest X value in the table,
		//we can't interpolate.
		if(iter == _points.cbegin() and x <= _points.cbegin()->first) {
			return _points.cbegin()->second;
		}

		//We can interpolate!
		float upperX{iter->first};
		float upperY{iter->second};
		float lowerX{(iter - 1)->first};
		float lowerY{(iter - 1)->second};

		float deltaY{upperY - lowerY};
		float deltaX{upperX - lowerX};

		return lowerY + ((x - lowerX)/ deltaX) * deltaY;
	}

private:
	//Our container of (x,y) data points
	//std::pair::<float, float>.first = x value
	//std::pair::<float, float>.second = y value
	std::vector<std::pair<float, float>> _points;
};

#endif //CUCBET_INTERPOLATOR_CUH
