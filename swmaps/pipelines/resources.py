# swmaps/pipelines/resources.py
from dagster import ResourceDefinition

from swmaps.config import settings

data_root_res = ResourceDefinition.hardcoded_resource(settings.data_root)
