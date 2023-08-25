import data_generation.motifs as motifs
import data_generation.custom_dataset as custom_dataset
from data_generation import dataset_wrappers
from poolblocks import perturbing_distributions

__all__ = [motifs.HouseMotif, motifs.FullyConnectedMotif, motifs.BinaryTreeMotif, motifs.CircleMotif, motifs.SetMotif,
           motifs.SplitHexagon, motifs.CrossHexagon, motifs.ReplicationMotif, motifs.IntermediateNodeMotif,

           custom_dataset.UniqueMotifCategorizationDataset,
           custom_dataset.UniqueMultipleOccurrencesMotifCategorizationDataset,
           custom_dataset.UniqueHierarchicalMotifDataset,
           custom_dataset.SimpleMotifCategorizationDataset,

           dataset_wrappers.CustomDatasetWrapper,
           dataset_wrappers.ExpressivityWrapper,
           dataset_wrappers.TUDatasetWrapper,
           dataset_wrappers.MutagWrapper,
           dataset_wrappers.MutagenicityWrapper,
           dataset_wrappers.RedditBinaryWrapper,
           dataset_wrappers.EnzymesWrapper,
           dataset_wrappers.PtFileWrapper,
           dataset_wrappers.BBBPWrapper,
           dataset_wrappers.BBBPAtomWrapper,

           perturbing_distributions.GaussianPerturbation]

from data_generation.serializer import ArgSerializable

def _from_dict_obj(o):
    if isinstance(o, dict) and "_type" in o:
        obj_class = next((x for x in __all__ if x.__name__ == o["_type"]), None)
        if obj_class is None:
            raise ValueError(f"Could not find class named {o['_type']}!")
        kwargs = {k: _from_dict_obj(v) for k, v in o["args"].items()}
        return obj_class(**kwargs)
    elif isinstance(o, list):
        return [_from_dict_obj(i) for i in o]
    else:
        return o

def from_dict(dict_repr: dict) -> ArgSerializable:
    return _from_dict_obj(dict_repr)