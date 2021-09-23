# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
from subprocess import run
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.data import parsers
from alphafold.common import residue_constants
from alphafold.data.tools import hhblits, hmmsearch
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer, param
import numpy as np


FeatureDict = Mapping[str, np.ndarray]

def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(sequence)
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = int_msa
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features

class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: Optional[str],
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               # pdb_database_path: Optional[str],
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               jackhmmer_param = None,
               hmmsearch_param = None,
               hhsearch_param = None,
               hhblits_param = None,
               customdb_jackhmmer = [],
               customdb_hhblits = [],
               customdb_max_hit: int = 500):

    """Constructs a feature dict for a given FASTA file."""
    if uniref90_database_path is not None:
      self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=uniref90_database_path, param=jackhmmer_param)
    else:
      self.jackhmmer_uniref90_runner = None

    if bfd_database_path is not None and uniclust30_database_path is not None:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path],param=hhblits_param)
    else:
      self.hhblits_bfd_uniclust_runner = None

    if mgnify_database_path is not None:
      self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=mgnify_database_path,param = jackhmmer_param)
    else:
      self.jackhmmer_mgnify_runner = None

    self.customjackhmmer_runners = []
    self.customhmmblits_runners = []

    for i in customdb_jackhmmer:
      self.customjackhmmer_runners.append(jackhmmer.Jackhmmer(binary_path=jackhmmer_binary_path, database_path=i, param=jackhmmer_param))

    for i in customdb_hhblits:
      self.customhmmblits_runners.append(hhblits.HHBlits(binary_path=hhblits_binary_path, databases=i, param=hhblits_param))
    #self.hhsearch_pdb70_runner = hhsearch.HHSearch(
    #    binary_path=hhsearch_binary_path,
    #    databases=[pdb_database_path],param=hhsearch_param)
    #if pdb_database_path is None:
    #  self.hhsearch_pdb70_runner = None

    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.customdb_max_hit = customdb_max_hit

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)


    uniref90_msa = uniref90_deletion_matrix = None
    if self.jackhmmer_uniref90_runner is not None:
      jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(input_fasta_path)[0]
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(jackhmmer_uniref90_result['sto'], max_sequences=self.uniref_max_hits)
      uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
      with open(uniref90_out_path, 'w') as f:
        f.write(jackhmmer_uniref90_result['sto'])
        
      uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
      logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    
    
    mgnify_msa = mgnify_deletion_matrix = None
    if (self.jackhmmer_mgnify_runner is not None):
      jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(input_fasta_path)[0]
      mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
      with open(mgnify_out_path, 'w') as f:
        f.write(jackhmmer_mgnify_result['sto'])
      
      mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
      mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
      mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))


    results = []
    jackhmmer_msas = []
    jackhmmer_deletion_mats = []
    count = 0
    for runner in self.customjackhmmer_runners:
      results.append(runner.query(input_fasta_path)[0])
      outpath = os.path.join(msa_output_dir, 'custom_jackhmmer'+str(count)+'.sto')
      with open(outpath, 'w') as f:
        f.write(results[-1]['sto'])
      tt, ttt = parsers.parse_stockholm(results[-1]['sto'])
      tt = tt[:self.customdb_max_hit]
      ttt = ttt[:self.customdb_max_hit]
      jackhmmer_msas.append(tt)
      jackhmmer_deletion_mats.append(ttt)
      count += 1


    #hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
    #pdb_out_path = os.path.join(msa_output_dir, 'pdb_hits.hhr')
    #with open(pdb_out_path, 'w') as f:
    #  f.write(hhsearch_result)

    #hhsearch_hits = parsers.parse_hhr(hhsearch_result)
    
    bfd_msa = bfd_deletion_matrix = None
    if self.hhblits_bfd_uniclust_runner is not None:
      hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(input_fasta_path)
      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
      with open(bfd_out_path, 'w') as f:
        f.write(hhblits_bfd_uniclust_result['a3m'])
      bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))


    results = []
    hhblits_msas = []
    hhblits_deletion_mats = []
    count = 0
    for runner in self.customhmmblits_runners:
      results.append(runner.query(input_fasta_path))
      outpath = os.path.join(msa_output_dir, 'hhblits_custom' + str(count) + '.a3m')
      with open(outpath, 'w') as f:
        f.write(results[-1]['a3m'])
      tt, ttt = parsers.parse_a3m(results[-1]['a3m'])
      hhblits_msas.append(tt)
      hhblits_deletion_mats.append(ttt)
      count += 1


    t1 = []
    t2 = []

    t1.extend(hhblits_msas)
    t1.extend(jackhmmer_msas)
    t2.extend(hhblits_deletion_mats)
    t2.extend(jackhmmer_deletion_mats)
    
    if uniref90_msa is not None:
      t1.append(uniref90_msa)
      t2.append(uniref90_deletion_matrix)
    if bfd_msa is not None:
      t1.append(bfd_msa)
      t2.append(bfd_deletion_matrix)
    if mgnify_msa is not None:
      t1.append(mgnify_msa)
      t2.append(mgnify_deletion_matrix)

    msa_features = make_msa_features(
        msas=t1,
        deletion_matrices=t2)

    
    
    
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    
    return msa_features
