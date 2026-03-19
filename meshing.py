from pathlib import Path
import gmsh
import math



#-------------------------------------------------------------------
def create_cube_mesh_p(
    filepath: Path,
    width: float,
    height: float,
    thickness: float,
    mesh_size: float,
    target_rw: float, # Added target_rw for explicit control
    center_z: float = 0.0,
    debug_gui: bool = False,
    local_ref=True,
) -> None:
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
   
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0 if local_ref else mesh_size) 
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0) 
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    gmsh.model.add(filepath.stem)
    
    z0 = center_z - thickness/2.0
    z1 = center_z + thickness/2.0

    coords = [
        (0.0,     0.0,     z0),  
        (width,   0.0,     z0),  
        (width,   height,  z0),  
        (0.0,     height,  z0),  
        (0.0,     0.0,     z1),  
        (width,   0.0,     z1),  
        (width,   height,  z1),  
        (0.0,     height,  z1)   
    ]

    p_center = gmsh.model.occ.addPoint(width/2, height/2, center_z, target_rw) 
    pts = [gmsh.model.occ.addPoint(x, y, z, mesh_size) for x, y, z in coords]
    gmsh.model.occ.synchronize()

    l1  = gmsh.model.occ.addLine(pts[0], pts[1])
    l2  = gmsh.model.occ.addLine(pts[1], pts[2])
    l3  = gmsh.model.occ.addLine(pts[2], pts[3])
    l4  = gmsh.model.occ.addLine(pts[3], pts[0])
    l5  = gmsh.model.occ.addLine(pts[4], pts[5])
    l6  = gmsh.model.occ.addLine(pts[5], pts[6])
    l7  = gmsh.model.occ.addLine(pts[6], pts[7])
    l8  = gmsh.model.occ.addLine(pts[7], pts[4])
    l9  = gmsh.model.occ.addLine(pts[0], pts[4])
    l10 = gmsh.model.occ.addLine(pts[1], pts[5])
    l11 = gmsh.model.occ.addLine(pts[2], pts[6])
    l12 = gmsh.model.occ.addLine(pts[3], pts[7])
    gmsh.model.occ.synchronize()

    cl_bot   = gmsh.model.occ.addCurveLoop([ l1,  l2,  l3,  l4])
    cl_top   = gmsh.model.occ.addCurveLoop([ l5,  l6,  l7,  l8])
    cl_back = gmsh.model.occ.addCurveLoop([ l1,  l10, -l5,  -l9])
    cl_front = gmsh.model.occ.addCurveLoop([-l3,  l11,  l7, -l12])
    cl_right  = gmsh.model.occ.addCurveLoop([ l9,  -l8, -l12,  l4])
    cl_left = gmsh.model.occ.addCurveLoop([ l10, l6,  -l11, -l2])
    gmsh.model.occ.synchronize()

    s_bot   = gmsh.model.occ.addPlaneSurface([cl_bot])
    s_top   = gmsh.model.occ.addPlaneSurface([cl_top])
    s_front = gmsh.model.occ.addPlaneSurface([cl_front])
    s_back  = gmsh.model.occ.addPlaneSurface([cl_back])
    s_left  = gmsh.model.occ.addPlaneSurface([cl_left])
    s_right = gmsh.model.occ.addPlaneSurface([cl_right])
    gmsh.model.occ.synchronize()

    sl  = gmsh.model.occ.addSurfaceLoop([s_bot, s_top, s_back, s_front, s_left, s_right])
    vol = gmsh.model.occ.addVolume([sl])
    gmsh.model.occ.synchronize()

    pg0 = gmsh.model.addPhysicalGroup(0, pts)
    gmsh.model.setPhysicalName(0, pg0, "points")

    all_edges = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]
    pg1 = gmsh.model.addPhysicalGroup(1, all_edges)
    gmsh.model.setPhysicalName(1, pg1, "edges")

    face_map = {
        "bottom": s_bot,
        "top":    s_top,
        "back":  s_back,
        "front":   s_front,
        "left":   s_left,
        "right":  s_right,
    }
    for name, surf_tag in face_map.items():
        pg = gmsh.model.addPhysicalGroup(2, [surf_tag])
        gmsh.model.setPhysicalName(2, pg, name)

    pgc = gmsh.model.addPhysicalGroup(0, [p_center]) 
    gmsh.model.setPhysicalName(0, pgc, "center")

    pg3 = gmsh.model.addPhysicalGroup(3, [vol])
    gmsh.model.setPhysicalName(3, pg3, "volume") 

#------------------------------------------------ Geometric refining
    if local_ref: 

        d_id = gmsh.model.mesh.field.add("Distance") 
        gmsh.model.mesh.field.setNumbers(d_id, "PointsList", [p_center])

        t_id = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_id, "InField", d_id)
        gmsh.model.mesh.field.setNumber(t_id, "SizeMin", target_rw)     # Size inside/at center
        gmsh.model.mesh.field.setNumber(t_id, "SizeMax", mesh_size)    # Size outside


        safe_dist_min = max(target_rw * 2, mesh_size * 0.1)
        size_ratio = mesh_size / target_rw
        growth_buffer = target_rw * size_ratio * 0.5 # Dynamic buffer
        safe_dist_max = max(safe_dist_min + growth_buffer, min(width, height, thickness)* 0.15)

        gmsh.model.mesh.field.setNumber(t_id, "DistMin", safe_dist_min) 
        gmsh.model.mesh.field.setNumber(t_id, "DistMax", safe_dist_max) 

        gmsh.model.mesh.field.setAsBackgroundMesh(t_id)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
#---------------------------------------------------
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.embed(0, [p_center], 3, vol)

    gmsh.model.mesh.generate(3)

    if debug_gui:
        # Launch the Gmsh GUI with the generated geometry
        # This will block the notebook until the GUI is closed
        print("Launching Gmsh GUI for geometry inspection...")
        gmsh.fltk.run()
        # Clean up Gmsh context but DO NOT proceed to meshing/writing
        gmsh.finalize()
        return

    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()


#-----------------------------------------------------------------