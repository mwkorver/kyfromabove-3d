"""The build's steps, each a module with its own command line: colour a
building from the frames (``colorize_building``), make its mesh
(``build_mesh``), check its texture (``check_building``), audit it
(``audit_building``), find its outline in the LiDAR (``lidar_outline``), the
Overture boxes (``overture_boxes``) and the publish step (``publish_step``);
and what they borrow: which frames see a wall (``wall_visibility``) and the
renderer it draws them with (``register_video``, ``render_flyaround``).

``pipeline.run_main`` calls a step's ``main`` in its own process; each
``scripts/<step>.py`` is the same command, run from a checkout.
"""
