#pragma once

#include <vector>
#include <cstdint>

/// Integer grid cell offset relative to the vehicle center.
struct CellOffset
{
    int dx;
    int dy;
};


///
/// Stores a rasterized footprint for a set of discrete orientations.
///
/// The lookup table is generated once during construction.
/// During planning, collision checking becomes only:
///
///     for(cell : lookup.orientation(idx))
///         grid[gx+cell.dx][gy+cell.dy]
///
class FootprintLookup
{
public:
    FootprintLookup(
        double length,
        double width,
        double grid_resolution,
        int angle_bins = 72);

    /// Returns the footprint corresponding to an orientation index.
    const std::vector<CellOffset>& orientation(int index) const;

    /// Converts an angle (rad) into the nearest lookup index.
    int angleIndex(double theta) const;

    int angleBins() const
    {
        return angle_bins_;
    }

    double resolution() const
    {
        return resolution_;
    }

private:
    //--------------------------------------------
    // lookup generation
    //--------------------------------------------
    void buildLookup();

    std::vector<CellOffset> rasterizeFootprint(
        double theta) const;

    //--------------------------------------------
    // geometry helpers
    //--------------------------------------------
    struct Point
    {
        double x;
        double y;
    };

    std::vector<Point> rectangleCorners(
        double theta) const;

    bool pointInsidePolygon(
        const Point& p,
        const std::vector<Point>& polygon) const;

    //--------------------------------------------
    // parameters
    //--------------------------------------------
    double length_;
    double width_;
    double resolution_;

    int angle_bins_;

    //--------------------------------------------
    // lookup table
    //--------------------------------------------
    std::vector<std::vector<CellOffset>> lookup_;
};