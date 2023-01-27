import sys
import os
import argparse
import view_surface
import combine_files

'''
Convenience program that allows to run the entire surface construction and
visualization pipeline, tarting with a set of randomly organized loss samples
stored in csv files and pre-computed learning trajectory projection and first
two principal components
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create visualization of loss surface from a set of csv files containing samples')

    parser.add_argument('-p', '--path', default='', type=str, help='Path to root of data directory')
    parser.add_argument('-t', '--trajectory', default='', help='File containing training trajectory')
    parser.add_argument('-o', '--output', required=True, help='Name of output file for surface and projected trajectory')
    parser.add_argument('--x', nargs=3, type=float, help='Range of x sampling: xmin xmax xnum')
    parser.add_argument('--y', nargs=3, type=float, help='Range of y sampling: ymin ymax ynum')
    parser.add_argument('--trim', action='store_true', help='Trim surface to specified x, y bounds')
    parser.add_argument('--show_edges', action='store_true', help='Show surface triangulation')
    parser.add_argument('--flatten', action='store_true', help='Create 2D representation of loss surface with isocontours')
    parser.add_argument('--skip', action='store_true', help='Skip surface creation if file already present')
    parser.add_argument('--show_steps', action='store_true', help='Show training steps')
    parser.add_argument('--traj_diameter', type=float, default=0.5, help='Diameter of trajectory tube representation')
    parser.add_argument('--step_diameter', type=int, default=None, help='Diameter of spherical step representation')
    parser.add_argument('--color_surface', action='store_true', help='Apply viridis color mapping to loss surface')
    parser.add_argument('--range', type=float, nargs=2, default=[0.,0.], help='Value range to consider for color mapping')
    parser.add_argument('--iso_diameter', type=float, default=3, help='Diameter of isocontours')
    parser.add_argument('--show_isovalues', action='store_true', help='Show loss values on level sets')
    parser.add_argument('--font_size', type=int, default=20, help='Font size for isocontour labels')
    parser.add_argument('--size', type=int, nargs=2, default=[1920, 1080], help='Window resolution')
    parser.add_argument('--traj_palette', type=str, default='Oranges', help='Palette to use to color code training steps')
    parser.add_argument('--show_colorbars', action='store_true', help='Display color bars')
    parser.add_argument('--frame_basename', type=str, default='frame', help='Basename of exported frames')
    parser.add_argument('--camera_basename', type=str, default='camera', help='Basename of camera settings export files')
    parser.add_argument('--camera_file', type=str, default='', help='File containing camera setting to use')
    parser.add_argument('--light_file', type=str, default='', nargs='+', help='File containing light information')
    parser.add_argument('--surf_name', type=str, help='Name of loss surface')
    parser.add_argument('--background', type=float, nargs=3, default=[0.,0.,0.], help='Window background color')

    args = parser.parse_args()
    print(args)

    if os.path.exists(args.output) and args.skip:
        print('{args.output} is alreay created. Skipping.')
    elif not args.trajectory:
        print('Missing trajectory information. Unable to proceed')
        sys.exit(0)
    else:
        combine_files.compute(args)

    args.path = ''
    args.surface = args.output
    for p in view_surface.parameters.keys():
        if p not in args:
            print(f'p is {p}')
            setattr(args, p, view_surface.parameters[p][0])
            print(f'{p} was missing and assigned {view_surface.parameters[p][0]} value')
    setattr(args, 'do_log', False)
    setattr(args, 'is_log', False)

    args.info = None
    base, ext = os.path.splitext(args.output)
    args.trajectory = base + "_trajectory" + ext

    view_surface.view(args)
