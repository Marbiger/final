# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.llm_interface import SpatialLLMInterface

def main():
    llm = SpatialLLMInterface('configs/llm_config.yaml')
    bbox1 = (0.4, 0.5, 0.2, 0.3)
    bbox2 = (0.6, 0.5, 0.2, 0.3)
    result = llm.predict_spatial_relation(bbox1, bbox2)
    print(f"Relationship: {result.relation_text}")
    print(f"Class ID: {result.relation_class}")

if __name__ == "__main__":
    main()