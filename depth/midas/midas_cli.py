
def cli(parser):
    group = parser.add_argument_group('Midas parameters')
    
    group.add_argument('-o', '--output_path', 
        default='output/',
        help='folder for output images'
    )

    group.add_argument('-m', '--model_weights', 
        default=None,
        help='path to the trained weights of model'
    )

    group.add_argument('-t', '--model_type', 
        default='midas_v21_small',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )

    group.add_argument('--optimize', dest='optimize', action='store_true')
    group.add_argument('--no-optimize', dest='optimize', action='store_false')
    group.set_defaults(optimize=True)

    