#ifndef POITRI_H_
#define POITRI_H_

#include "vec.h"

float point_segment_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, Vec3f &r);
float point_triangle_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, const Vec3f &x3, Vec3f &r);


#endif