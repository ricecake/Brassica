Bamboo for a good plant type
Trees on voronoi grid.  Even spacing, but not regular.

Make biome pallate cosine based.
project the froxel values, then apply a TAA
Apply a taa to clouds after lower res rendering.
Have a joint filter phase for clouds and fog to ensure secure.
Unified volumetric raymarching

Compute shadow volume for pre-built models?
C++ procedural music
Procedural waves
Leaves? Bark? Moss? Hair?
Split the lights into cpu lights and gpu lights, to avoid weird round trips.
Integrate the mood manager.

Tri-hex texturing for terrain texture breakup - try using the hex-wobble to make it more dynamic?  Maybe needless.
Need to drop sky distinction for auto-expose
Need bark.  More details on plants.

How to track weather at the camera?  If I want to override the weather at the camera, how do I do that in a way that doesn't create a wall?
Divide the world into a grid, weather spreads on the grid, when an override is entered have a noisy boundary ring spread from the camera coordinate outward to some set distance fading the basic primatives in the needed direction.  simplex grid, shallow fade.  No solid wall of "no clouds", they just get more sparse.  spread over a wide area, and center on the centroid of the current simplex, not the camera.  That way when the camera moves it isn't a sliding ring, but the shift can still fade and follow "kinda" closely.  maybe a 1-2 km simplex?

optical depth/scattering/transmittance via spherical harmonics?
describe media boundaries via Fourier stuff.
dynamically update bounding map - store coarse bounds in texture, and a reference to a chain of sub bounds.   - cell id is reference?

Figure out good way to handle shadows going upwards into clouds.

Divide the world into a grid of relatively fine precision.    Define positions relative to one of these grid points.  Adds one field to position in the form of a morton point id, keeps most rendering camera relative, while also allowing more precise anchor points that are still world static.

Make a trail that's a vine.  Since we're doing more mesh shaders, it should be possible to make something that looks like a flowery vine.
Tie distance from trunk to procedural items.  Have things grow and bloom.

When doing unified volumetric with participating media, if the ray gets to water backup before sampling to always get the exact surface.  Model rater as an sdf sphere so it's easy to know when there, and then offset surface by wave function.
Evaluate light at surface of water, use that to drive floating caustics in the water as part of underwater volumetric fog.

Do bounding map march.  Average angles for efficiency of pre compute.  Use different rows for bounding of different materials or segments. Is a texture the right structure?
Can a prefix sum be updated efficiently?
What about the bounding map?
March and accumulate segments, then integrate over the segments?

Use stencil to mark sky, land water to delimit atmosphere bounds

multiple terrain height sync regions: return a handle, and allow them to be added to or translated that way.  The goal is to be able to have an entity manager system have an array of entities as footprints, and to automatically sort and merge those footprints into efficient transfer regions with padding.  Maybe a metric of the ratio of the transfer region to the raw footprint area?

Take organic cell, and find centroid of values.  If pulled towards neighbor, vector from center towards centroid.
That's not possible.   Just vector from center to centroid, and find way to use to guide phasor noise.
4d phasor variant? Quaterniors?

Longest edge bisection of a triangle, random noise subdivision, proximity to edge intensifies color, probability of bisection increase with distance downward.   Makes lightning?

Crack things using the organic cell tool to divide the object, then biset deeper the closer it is to the impact/impact vector.

Periodic polynomial noise with multiple derivatives.  Track vector as complex components?  Might be good proxy for sine.   Can it track its own derivatives to somehow respond to it? Second derivative?

Unified volumetric rendering.  Gather optical properties , then composite for view.

add more properties to the terrain tectonic plates.  Give them angular velocity.
Give them soil and rock composition or 'softness'.  This can drive overall biome type, like dirt color and plant suitability.  Then use binary space partition to divide into sub-biomes and micro biomes.  This can be used to dictate that certain areas have a super-biome of one type, but still handle the physical properties correctly.

Cellular noise.  Simplex grid.  Circle at each vertex.   Circle generates a periodic function.  Evaluate at each edge.   Boundary is the average of the two values.     Each cell can have a phase and an offset, as well as custom terms.   Should be a way to efficiently solve for intersection.

Cell system for fractal boundary evaluation - eval noise with threshold. At boundary line keep subdividing
Same for sub regions like clearing in forest and such.  The boundary helps with blending biomes, or adding sub flavor.

Need the water table to be based on rough height, and then look at erosion depth.  Closer erosion gets to
Aggressively stagger biome update - 128 cells per frame
Fand a way to advect cloud volume during update , or increment phase
Make it so nodes are only given requested or declared resources.    Maybe it needs to specify how to deliver them?
Shadow volume?
Simplify biome to be closer to exactly what's needed
Add shader for fire
Lightning soon
Snow fall
Add water table, and lower gullies where closer.  Get water shader to represent small drops for flat water on a slope.
Explicit rain map
Render order for volume clouds atmosphere?
Lightning
Better rain and snow
Glint for snow.
Make tree proxy elevated.
Object shadows
A structure
Reverse z
Cloud sculpting
Terrain sculpting
City
Wind system as below
Mood engine port
Fix sun chasing and world map
Cache tree types in buffer, mix and match parts instead of computation
Smooth Worley for tree proxies.
Move clouds around terrain
Simplify algebra and forward differentiation
Cloud erosion
Simplify noise calls
Curl of cross noise
Fix biome blending
Cluster trees. Make lone trees less common
Big thing in orbit
Alternative to wind for realistic climate.   Cellular automata that sees height? Plants retain moisture
Unify volumetric and atmosphere systems
Reduce cloud complexity.   One texture, two at most. Independent weather maps.  Decimation based edges, lower res with temporal accumulation.  Don't update 3d volume.   Update the weather map very slowly.
Move variance into the the main terrain heightmap.
Make the biome map track sediment, ridge, geological biome, water table
Rework river system to feed from gullies
Make water table feed from noise.   Make generally higher in higher elevation places
Make shaders more proper objects.  Have them define their own push constants.   Have a shader subscribe to a list of features.   Automate packing the flags into a uint, and inject a function to extract per shader.

Make sure the resource tracking and queue system sees that a Host/Code node is able to produce or own resources.  That way things that populate buffers, or like the lights, can be defined code nodes that get scheduled to have a chance to run before their data is used.

All nodes are owned by a manager.  A manager owns any associated predetermined textures or resources.  Add a simple template node that explicitly handles syncing.
Now the graph is instantiated.  The service locator provides a set of all managers.  Each manager provides the nodes that it managers, and also a list of the resource nodes it's in control of.  Now when a node is instantiated, the graph can pass a reference to the manager  for it which will make it easier to check activation and such.  Managers own groups of nodes, loosely, so that it can be disabled and automatically propagate that to its nodes.
Allows dropping the registration macro from passes, makes activation more coordinated, and facilitates grouping passes along with the thing that controls them and manages the ui and  config for them.

Make a node I generic that shaders and a json file can define a pass.

Decrease size of voronoi cell as height increases, make river network

I would like to implement simple world wrap for the camera as it travels around the world, with the world dimensions wrapping in a fashion consistent with the surface area of the 600km radius planet that has been consistently referenced.
Crossing the east/west seam should move the camera to the otherside of the map without changing orientation of the camera. Moving the camera across the north/south seam should flip the camera orientation (North component becomes South Component) and wrap the camera to the matching "opposite" side of the same end of the map. Basic fake sphere wrapping on a grid.
The position of the sun and the moon should update as the camera moves according to an ephemeris model. The day should be hardcoded to be the summer equinox for the northern hemisphere.    The UI should display a utc equivalent centered around latitude and longitude zero, as well as camera local time.   The UI should display camera latitude and longitude.   The coordinate system should be centered such that xy 00 should match lat lon 00.



Want a biome map that's a lower res.
Want a visibility map that also bakes lod into the visibility. IE lod -1 is occluded.
Need to extend the lod system to support lod0 resolution at lod1, and so on.  This will cost texture, but will allow promoting a tile to higher lod for focus or detail.

Store resource desc data in the structure, and imply the operation by method used to add to realization.  Also include if it's a historical structure or persistent there.  Then the persistent data becomes much more easy to reason about.  Create means "source of fresh data", so anything that reads should want to come after creation.

Need to find a way for the deep beer maps to either always start at the top of the cloud layer, or to have it make up the difference with march.  Same for super distant terrain.

Use weather advection system to make global cloud coverage map.   Density, type, lighting, extinction.    Fill two buffers, and sample from one and slowly merge in the other.  When they're merged repopulate the first with data from weather simulation and repeat.   Should allow slow update.

Material system stores palate, and type for procedural corolr? Or specified how to apply biome colors? Blend with biome?
Type can specify solid colors, which field varies, procedural noise, any way of specifying color and material properties.   Should be able to specify the different properties using different approaches, so flat roughness, worley metallic, etc.    Need to find a way to specify how it interpolates.   That might be an issue for the thing laying out the materials.   But all of them should be able to blend.   Maybe that's not possible?

Look into using stencils to drive shading variation for water, clouds, sky and land.  By having dynamic stencil state, I can use nearly the same shader for each (tone mapping) but have different basic tone mapping properties.  Makes the waterline problem a lot easier.

Use gradient of cloud noise to drive awareness of need for detail noise? Proxy for ambient occlusion?

Some manner of template node, or a way to say that we do the same thing to multiple things, but specialize how.

lightup plants - lightup underwater stuff.  -- MORE LASERS - GLITTER - CAT BUTTS -- MEEEECHS

On the fly cloud lightning by pupolating a thin, wispy noise texture attached loosely to cloud density and moisture.

Make beer opacity maps sample clouds and also volumetric.     Unify sizes.

Find plant morphology terms and ranges and model them in plants procedurally.

Port the procedural system for geometry generation.

Deep opacity maps for shadow stuff
Cull terrain that cannot possibly cast a shadow by evaluating shadow only for terrain that's not facing the sun - if you're facing the sun, you cannot cast a shadow, only receive.

For constraint system, find the first and second derivative of relationships.   Will tell you how to tweak to get change and how responsive it is to the change.

Embrace the terrain wrapping.  Since the world is infinite, but wraps around, make it so you can move up and down between worlds.  Essentially blending the heightmaps for a word circumferance away into the current one.  Makes for a weird sort of fast travel.

Make a average temperature chart based on altitude, an latitude , rough weather patterns.  Very low resolution.  One texture for the planet.
Make areas facing the wind have higher moisture.  Make proximity to ocean have higher water.

Adapt the sky tone mapping to work like the tiered asc CDL settings.  Have the ability to have different tone mapping, exposure and bloom settings based on distance parameters.

Make shadows and gtao possible
Ray mapped occlusion - cone occlusion?
Have weather map distort water layer
Bring over the rest of the particles and lighting clusters
Bring over the grass and fern
Activate async compute shaders

make the clipmap system readback the clipmap array, or be able to, for queries around height.

Compute a spatial voronoi where distance is attenuation and normal based.   For probes of light

For noise use phase and varying frequency to stretch the pattern over the planet.   Then it fits with camera relative render because it's just a lookup.   Then for close-up noise it doesn't matter if it repeats a bit more often since it's tiled on top of the other, and smaller.


Fixup biome map
How to antialias better?
Need stars to show up
Need space view better
Moving should be able to "chase" sun.
Underwater atmosphere scattering
Real lighting
Fix shadows
Implement ambient
Registry of color palette
Little particle fish.
Little particle balloons
Birds
Mesh transfer
Flow map
Simulated hydrology.

Steal suckerpunch vorticle notion, which is a non-rendering particle that can affect the wind of other particles.  Sphere with strength, radius, orientation, vector.  Vector is on the position of the sphere and allows localized wind.

Make wind a simplex noise for intensity.  Have a 2d vector for direction, and a 2d offset vector.  The offset vector lets the wind change direction without weird artifacts.  But calculate advection as time * direction vector.  On change, save offset+time*vector -> offset.  That way we don't need to track explicit history or complicated advection.

Psrd noise with shifting phase, advecting with a slightly faster voronoi shell for zelda style magma.

Copy the WeatherAttribute system with interpolation, as well as the spring dampening stuff, Copy the biome interpolation stuff.  Both of these are good things to make generic.  Also the mood engine, and the constraint enforcement system for the weather that finds the minimal nudge.    Move to modeling a lot of those traits as a grid of connected springs, so that values pull each other around according to variations.  Cheaper than lbm, but still leaves connection in place.

-----

I would like the world coordinates to wrap at the edges, so that if enough distance is covered the terrain will wrap back around, like a sphere.

Mapping a 2d square to a sphere will never be perfect but I'd like it to be minimized, and for the blending to make it even to happen on the opposite side of the sphere from where the camera starts.

----

I would like pressing the '-' key to enable a first person camera mode.
This should put the camera about 3 units away from the ground and cap speeds to "superhero" levels of standard human. ctrl should make the camera move at a brisk walking speed and lower its height, like crouching. Shift should make it a hard sprint. Normal w forward should be a standard run. Pressing spacebar should trigger a jump. Gravity should be treated as normal earth gravity, and the camera should fall in a normal physical manner when not within distance of the ground.
For this, "superhero" should mean roughly twice that of a peak elite human. Running should be about twice the speed of peak distance running, sprinting twice the fastest sprint speed, and so on for jumping.
If the camera is underwater it should ignore gravity and move at similar speeds for swimming.
The camera should follow the acceleration and fov change system the camera already has.

