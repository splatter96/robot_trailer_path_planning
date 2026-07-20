#include "footprint_lookup.hpp"

#include <cmath>
#include <algorithm>

namespace
{
constexpr double TWO_PI = 2.0 * M_PI;
}

FootprintLookup::FootprintLookup(
    double length,
    double width,
    double resolution,
    int angle_bins)
    :
    length_(length),
    width_(width),
    resolution_(resolution),
    angle_bins_(angle_bins)
{
    lookup_.resize(angle_bins_);

    buildLookup();
}

const std::vector<CellOffset>&
FootprintLookup::orientation(int index) const {
    return lookup_[index];
}

void FootprintLookup::buildLookup() {
    const double angle_step = TWO_PI / static_cast<double>(angle_bins_);

    for (int i = 0; i < angle_bins_; ++i) {
        double theta = i * angle_step;
        lookup_[i] = rasterizeFootprint(theta);
    }
}

std::vector<CellOffset>
FootprintLookup::rasterizeFootprint(double theta) const {
    const auto polygon = rectangleCorners(theta);

    double min_x = polygon.front().x;
    double max_x = polygon.front().x;
    double min_y = polygon.front().y;
    double max_y = polygon.front().y;

    for (const auto& p : polygon) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);

        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    const int gx_min = static_cast<int>(std::floor(min_x / resolution_));
    const int gx_max = static_cast<int>(std::ceil(max_x / resolution_));
    const int gy_min = static_cast<int>(std::floor(min_y / resolution_));
    const int gy_max = static_cast<int>(std::ceil(max_y / resolution_));

    std::vector<CellOffset> cells;

    cells.reserve( (gx_max - gx_min + 1) *
                   (gy_max - gy_min + 1));

    for (int gx = gx_min; gx <= gx_max; ++gx) {
        for (int gy = gy_min; gy <= gy_max; ++gy) {
            Point center;
            center.x = (gx + 0.5) * resolution_;
            center.y = (gy + 0.5) * resolution_;

            if (pointInsidePolygon(center, polygon)) {
                cells.push_back({gx, gy});
            }
        }
    }

    std::sort(
        cells.begin(),
        cells.end(),
        [](const CellOffset& a,
           const CellOffset& b)
        {
            const int da =
                a.dx * a.dx +
                a.dy * a.dy;

            const int db =
                b.dx * b.dx +
                b.dy * b.dy;

            return da < db;
        });

    return cells;
}

std::vector<FootprintLookup::Point>
FootprintLookup::rectangleCorners(double theta) const {
    const double half_l = length_ * 0.5;
    const double half_w = width_ * 0.5;

    const double c = std::cos(theta);
    const double s = std::sin(theta);

    std::vector<Point> corners(4);

    auto rotate = [&](double x, double y) {
        return Point{
            c * x - s * y,
            s * x + c * y
        };
    };

    corners[0] = rotate(-half_l, -half_w);
    corners[1] = rotate( half_l, -half_w);
    corners[2] = rotate( half_l,  half_w);
    corners[3] = rotate(-half_l,  half_w);

    return corners;
}

int FootprintLookup::angleIndex(double theta) const {
    theta = std::fmod(theta, TWO_PI);

    if (theta < 0.0)
        theta += TWO_PI;

    int idx = static_cast<int>(std::round(theta / TWO_PI * angle_bins_));

    return idx % angle_bins_;
}

bool FootprintLookup::pointInsidePolygon( const Point& p, const std::vector<Point>& polygon) const {
    bool inside = false;

    const size_t n = polygon.size();

    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const Point& pi = polygon[i];
        const Point& pj = polygon[j];

        const bool intersect =
            ((pi.y > p.y) != (pj.y > p.y)) &&
            (p.x < (pj.x - pi.x) *
                        (p.y - pi.y) /
                        (pj.y - pi.y + 1e-12)
                    + pi.x);

        if (intersect)
            inside = !inside;
    }

    return inside;
}
