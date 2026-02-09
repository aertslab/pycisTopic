import anndata as ad

def convert_to_cell_topic_adata(cistopic_obj, keep_projections=True, save_path=None):
    # convert cell_topic matrix and annotations
    cell_topic = cistopic_obj.selected_model.cell_topic.T
    cell_obs = cistopic_obj.cell_data
    cell_obs.fillna('str_nan', inplace=True)
    obsm_dict = {}
    if keep_projections and cistopic_obj.projections['cell']:
        for key in cistopic_obj.projections['cell'].keys():
            obsm_dict[f'X_{key.lower()}'] = cistopic_obj.projections['cell'][key].to_numpy()
    cell_topic_adata = ad.AnnData(X=cell_topic,
                                  obs=cell_obs,
                                  obsm=obsm_dict)
    
    if save_path is not None:
        cell_topic_adata.write_h5ad(save_path)
    
    return cell_topic_adata

def convert_to_region_topic_adata(cistopic_obj, keep_projections=True, save_path=None):
    # convert cell_topic matrix and annotations
    region_topic = cistopic_obj.selected_model.region_topic
    var_obs = cistopic_obj.region_data
    var_obs.fillna('str_nan', inplace=True)
    obsm_dict = {}
    if keep_projections and cistopic_obj.projections['region']:
        for key in cistopic_obj.projections['region'].keys():
            obsm_dict[f'X_{key.lower()}'] = cistopic_obj.projections['region'][key].to_numpy()
    region_topic_adata = ad.AnnData(X=region_topic,
                                  obs=var_obs,
                                  obsm=obsm_dict)
    
    if save_path is not None:
        region_topic_adata.write_h5ad(save_path)
    
    return region_topic_adata

def convert_to_adata(cistopic_obj, keep_projections=True, save_path=None):
    binary_matrix = cistopic_obj.binary_matrix.T
    fragment_matrix = cistopic_obj.fragment_matrix.T
    cell_obs = cistopic_obj.cell_data
    var_obs = cistopic_obj.region_data
    var_obs.fillna('str_nan', inplace=True)
    cell_obs.fillna('str_nan', inplace=True)
    path_to_fragments = cistopic_obj.path_to_fragments
    obsm_dict = {}
    if keep_projections:
        if cistopic_obj.projections['region']:
            for key in cistopic_obj.projections['region'].keys():
                obsm_dict[f'X_{key.lower()}'] = cistopic_obj.projections['region'][key].to_numpy()
        if cistopic_obj.projections['cell']:
            for key in cistopic_obj.projections['cell'].keys():
                obsm_dict[f'X_{key.lower()}'] = cistopic_obj.projections['cell'][key].to_numpy()
    adata = ad.AnnData(X=binary_matrix,
                                  obs=cell_obs,
                                  var=var_obs,
                                  obsm=obsm_dict,
                                  layers={'fragments': fragment_matrix},
                                  uns={'path_to_fragments': path_to_fragments})
    if save_path is not None:
        adata.write_h5ad(save_path)
    return adata

def run_convert(args):
    if args.conversion_type == 'cell_topic':
        convert_to_cell_topic_adata(args.cistopic_object, save_path=args.output_adata)
    elif args.conversion_type == 'region_topic':
        convert_to_region_topic_adata(args.cistopic_object, save_path=args.output_adata)
    elif args.conversion_type == 'cellxpeak':
        convert_to_adata(args.cistopic_object, save_path=args.output_adata)
    else:
        raise ValueError(f"Invalid conversion type: {args.conversion_type}. Please choose from 'cell_topic', 'region_topic', or 'cellxpeak'.")
    
def add_parser_convert(subparsers):
    """Creates an ArgumentParser to read the options for this script."""
    parser_convert = subparsers.add_parser(
        "convert",
        help="Convert old cistopic objects to adata files"
    )

    parser_convert.add_argument(
        "-i",
        "--cistopic_object",
        dest="cistopic_object",
        action="store",
        type=str,
        required=True,
        help="Path to the cistopic object file to be converted."
    )
    
    parser_convert.add_argument(
        "-o",
        "--output_adata",
        dest="output_adata",
        action="store",
        type=str,
        required=True,
        help="Path to the output .h5ad file.",
    )
    
    parser_convert.add_argument(
        "-t",
        "--type",
        dest="conversion_type",
        action="store",
        type=str,
        required=True,
        help="Type of conversion to perform (option: 'cell_topic', 'region_topic', 'cellxpeak')."
    )
    
    parser_convert.set_defaults(
        func=run_convert
    )