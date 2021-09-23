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


import json
import os
import pathlib
import pickle
import random
import sys
from typing import Dict

from six.moves import cStringIO

from absl import app
from absl import flags
from absl import logging
from alphafold.data import pipeline
from alphafold.data.tools import hhblits, hmmsearch, jackhmmer, param


flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', '/usr/bin/jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', '/usr/bin/hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', '/usr/bin/hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits, used with UniClust30.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits, used with BFD.')
# flags.DEFINE_string('pdb_database_path', None, 'Path to the PDB70 database for use by HHsearch.')
flags.DEFINE_list('custom_database_paths_jackhmmer', None, 'Paths to the custom database for msa build with jachmmer, seperated by commas')
flags.DEFINE_list('custom_database_paths_hhblits', None, 'Paths to the custom database for msa build with hhblits, seperated by commas')
flags.DEFINE_integer('custom_database_maxhit', 500, 'maxium MSA hit for custom database')

#======jackhmmer param =========
flags.DEFINE_integer('jackhmmer_ncpu', 8, '# of CPU to use with jackhmmer')
flags.DEFINE_integer('jackhmmer_niter', 1, '# of iter to use with jackhmmer')
flags.DEFINE_float('jackhmmer_evalue', 0.0001, 'Evalue to use with jackhmmer')
flags.DEFINE_integer('jackhmmer_zvalue', None, 'Zvalue to use with jackhmmer')
flags.DEFINE_float('jackhmmer_F1', 0.0005, 'F1 filter to use with jackhmmer')
flags.DEFINE_float('jackhmmer_F2', 0.00005, 'F2 filter to use with jackhmmer')
flags.DEFINE_float('jackhmmer_F3', 0.0000005, 'F3 filter to use with jackhmmer')
flags.DEFINE_float('jackhmmer_incdom_e', None, 'incdom_e to use with jackhmmer')
flags.DEFINE_float('jackhmmer_dom_e', None, 'incdom_e to use with jackhmmer')
flags.DEFINE_string('jackhmmer_flags', None, "flags using with jackhmmer")
#======hmmsearch flags =========
flags.DEFINE_integer('hmmsearch_ncpu', 8, '# of CPU to use with hmmsearch')
flags.DEFINE_float('hmmsearch_evalue', 0.0001, 'Evalue to use with hmmsearch')
flags.DEFINE_integer('hmmsearch_zvalue', None, 'Zvalue to use with hmmsearch')
flags.DEFINE_float('hmmsearch_F1', 0.0005, 'F1 filter to use with hmmsearch')
flags.DEFINE_float('hmmsearch_F2', 0.00005, 'F2 filter to use with hmmsearch')
flags.DEFINE_float('hmmsearch_F3', 0.0000005, 'F3 filter to use with hmmsearch')
flags.DEFINE_float('hmmsearch_incdom_e', None, 'incdom_e to use with hmmsearch')
flags.DEFINE_float('hmmsearch_dom_e', None, 'incdom_e to use with hmmsearch')
flags.DEFINE_string('hmmsearch_flags', None, "flags using with hmmsearch")
#======hhblitz flags ===========
flags.DEFINE_integer('hhblits_ncpu', 8, '# of CPU to use with hhblits')
flags.DEFINE_integer('hhblits_niter', 3, '# of iter to use with hhblits')
flags.DEFINE_float('hhblits_evalue', 0.001, 'Evalue to use with hhblits')
flags.DEFINE_integer('hhblits_maxseq', 1_000_000, 'maxseq to use with hhblits')
flags.DEFINE_integer('hhblits_realign_max', 100_000, 'realign_max to use with hhblits')
flags.DEFINE_integer('hhblits_maxfilt', 100_000, 'maxfilt to use with hhblits')
flags.DEFINE_integer('hhblits_min_prefilter_hits', 1000, 'min_prefilter_hits to use with hhblits')
flags.DEFINE_integer('hhblits_p', 20, 'p to use with hhblits')
flags.DEFINE_integer('hhblits_z', 500, 'z to use with hhblits')
flags.DEFINE_string('hhblits_flags', None, "flags using with hhblits")
#=======hhsearch flags ==========
flags.DEFINE_integer('hhsearch_ncpu', 8, '# of CPU to use with hhsearch')
flags.DEFINE_integer('hhsearch_niter', 3, '# of iter to use with hhsearch')
flags.DEFINE_float('hhsearch_evalue', 0.001, 'Evalue to use with hhsearch')
flags.DEFINE_integer('hhsearch_maxseq', 1_000_000, 'maxseq to use with hhsearch')
flags.DEFINE_integer('hhsearch_realign_max', 100_000, 'realign_max to use with hhsearch')
flags.DEFINE_integer('hhsearch_p', 20, 'p to use with hhsearch')
flags.DEFINE_integer('hhsearch_z', 500, 'z to use with hhsearch')
flags.DEFINE_string('hhsearch_flags', None, "flags using with hhsearch")

FLAGS = flags.FLAGS




def _check_flag(flag_name: str, preset: str, should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')


def build_msa(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline):

  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)


  feature_dict = data_pipeline.process(input_fasta_path=fasta_path, msa_output_dir=msa_output_dir)

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  #create param class for msa tools
  hhblits_param = param.HHBlitsParam(
    n_cpu=FLAGS.hhblits_ncpu,
    n_iter=FLAGS.hhblits_niter,
    e_value=FLAGS.hhblits_evalue,
    maxseq=FLAGS.hhblits_maxseq,
    realign_max=FLAGS.hhblits_realign_max,
    maxfilt=FLAGS.hhblits_maxfilt,
    min_prefilter_hits=FLAGS.hhblits_min_prefilter_hits,
    p=FLAGS.hhblits_p,
    z=FLAGS.hhblits_z,
    flags=FLAGS.hhblits_flags
  )

  hhsearch_param = param.HHSearchParam(
    n_cpu=FLAGS.hhsearch_ncpu,
    n_iter=FLAGS.hhsearch_niter,
    e_value=FLAGS.hhsearch_evalue,
    maxseq=FLAGS.hhsearch_maxseq,
    realign_max=FLAGS.hhsearch_realign_max,
    p=FLAGS.hhsearch_p,
    z=FLAGS.hhsearch_z,
    flags=FLAGS.hhsearch_flags
  )

  hmmsearch_param = param.HmmSearchParam(
    flags = FLAGS.hmmsearch_flags,
    n_cpu = FLAGS.hmmsearch_ncpu,
    e_value = FLAGS.hmmsearch_evalue,
    z_value = FLAGS.hmmsearch_zvalue,
    filter_f1 = FLAGS.hmmsearch_F1,
    filter_f2 = FLAGS.hmmsearch_F2,
    filter_f3 = FLAGS.hmmsearch_F3,
    incdom_e = FLAGS.hmmsearch_incdom_e,
    dom_e = FLAGS.hmmsearch_dom_e
  )

  jackhmmer_param = param.JackHmmerParam(
    n_cpu = FLAGS.jackhmmer_ncpu,
    n_iter = FLAGS.jackhmmer_niter,
    e_value = FLAGS.jackhmmer_evalue,
    z_value = FLAGS.jackhmmer_zvalue,
    filter_f1 = FLAGS.jackhmmer_F1,
    filter_f2 = FLAGS.jackhmmer_F2,
    filter_f3 = FLAGS.jackhmmer_F3,
    incdom_e = FLAGS.jackhmmer_incdom_e,
    dom_e = FLAGS.jackhmmer_dom_e,
    flags = FLAGS.jackhmmer_flags
  )

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  custom_db_jackhmmer = FLAGS.custom_database_paths_jackhmmer
  custom_db_hhblits = FLAGS.custom_database_paths_hhblits

  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      jackhmmer_param=jackhmmer_param,
      hhblits_param=hhblits_param,
      hhsearch_param= hhsearch_param,
      hmmsearch_param= hmmsearch_param,
      customdb_hhblits=custom_db_hhblits,
      custom_db_jackhmmer = custom_db_jackhmmer,
      customdb_max_hit = FLAGS.custom_database_maxhit)

  random_seed = random.randrange(sys.maxsize)
  

  # build msa for each fasta
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
    build_msa(fasta_path=fasta_path,fasta_name=fasta_name,output_dir_base=FLAGS.output_dir,data_pipeline=data_pipeline)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
  ])

  app.run(main)
