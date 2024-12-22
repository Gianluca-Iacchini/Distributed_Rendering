#ifndef RAYTRACINGUTILS_HLSLI
#define RAYTRACINGUTILS_HLSLI

bool RayAABBIntersectionTest(float3 rayOrigin, float3 rayDirection, float3 aabb[2], out float tmin, out float tmax)
{
    float3 tmin3, tmax3;
    int3 sign3 = rayDirection > 0;

    // Handle rays parallel to any x|y|z slabs of the AABB.
    // If a ray is within the parallel slabs, 
    //  the tmin, tmax will get set to -inf and +inf
    //  which will get ignored on tmin/tmax = max/min.
    // If a ray is outside the parallel slabs, -inf/+inf will
    //  make tmax > tmin fail (i.e. no intersection).
    // TODO: handle cases where ray origin is within a slab 
    //  that a ray direction is parallel to. In that case
    //  0 * INF => NaN
    const float FLT_INFINITY = 1.#INF;
    float3 invRayDirection = rayDirection != 0 ? 1 / rayDirection : (rayDirection > 0 ? FLT_INFINITY : -FLT_INFINITY);

    tmin3.x = (aabb[1 - sign3.x].x - rayOrigin.x) * invRayDirection.x;
    tmax3.x = (aabb[sign3.x].x - rayOrigin.x) * invRayDirection.x;

    tmin3.y = (aabb[1 - sign3.y].y - rayOrigin.y) * invRayDirection.y;
    tmax3.y = (aabb[sign3.y].y - rayOrigin.y) * invRayDirection.y;
    
    tmin3.z = (aabb[1 - sign3.z].z - rayOrigin.z) * invRayDirection.z;
    tmax3.z = (aabb[sign3.z].z - rayOrigin.z) * invRayDirection.z;
    
    tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
    tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
    
    return tmax > tmin && tmax >= RayTMin() && tmin <= RayTCurrent();
}

#endif // RAYTRACINGUTILS_HLSLI