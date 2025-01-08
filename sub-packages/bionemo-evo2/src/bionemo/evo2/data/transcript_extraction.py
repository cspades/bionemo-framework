import argparse
from collections import defaultdict
import math
import re
import sys
from bionemo.noodles import back_transcribe_sequence, complement_sequence, reverse_sequence, transcribe_sequence
from bionemo.noodles.nvfaidx import NvFaidx
from nemo.utils import logging

def parse_gtf_attributes(attributes: str):
    # Split on all semicolons that are not inside quotes
    attributes = re.split(r';(?=(?:[^"]*"[^"]*")*[^"]*$)', attributes)
    out = dict()
    for a in attributes:
        if len(a) == 0:
            continue
        key = a.split()[0]
        value = a.split('"')[1]
        out[key] = value
    return out

def extract_transcript_exons(gtf_path: str, only_longest_transcript: bool):

    genes = defaultdict(set)
    gene2transcripts = defaultdict(set)
    transcripts = dict()
    exons = dict()
    exon2transcript = dict()
    transcript2gene = dict()
    transcript2exon = defaultdict(set)
    skip_transcripts = set()

    gtf_fields = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    with open(gtf_path) as infile:
        for line in infile:
            # skip header lines
            if line.startswith("#"): continue
            line = line.strip().split("\t")
            if len(line) < 9:
                continue

            # parse the attributes into a dictionary
            line = dict(zip(gtf_fields, line))
            attribs = parse_gtf_attributes(line['attribute'])

            if line['feature'] == 'gene':
                contig, start, end, strand = line['seqname'], line['start'], line['end'], line['strand']
                start, end = int(line['start'])-1, int(line['end'])
                try:
                    gene_id = attribs['gene_id']
                except:
                    continue
                genes[gene_id].add((contig, start, end, strand))

            elif line['feature'] == 'exon':
                contig, start, end, strand = line['seqname'], line['start'], line['end'], line['strand']
                start, end = int(line['start'])-1, int(line['end'])
                try:
                    gene_id = attribs['gene_id']
                except:
                    continue
                transcript_id = attribs['transcript_id']
                gene2transcripts[gene_id].add(transcript_id)

                # Skip exons that have already been handled and are likely errors
                if transcript_id in skip_transcripts:
                    continue
                exon_number = int(attribs['exon_number'])

                exon_id = (gene_id, transcript_id, exon_number)
                if exon_id in exons:
                    del exons[exon_id]
                    if transcript_id in transcripts:
                        del transcripts[transcript_id]
                    if transcript_id in transcript2exon:
                        del transcript2exon[transcript_id]
                    skip_transcripts.add(transcript_id)
                    continue

                exons[exon_id] = {"seqname":contig, "start":start, "end":end, "strand":strand}
                if exon_id in exon2transcript:
                    raise Exception("Exon Already Exists in exon2transcript")
                exon2transcript[exon_id] = transcript_id
                transcript2exon[transcript_id].add(exon_id)

            elif line['feature'] == 'transcript':
                contig, start, end, strand = line['seqname'], line['start'], line['end'], line['strand']
                start, end = int(line['start'])-1, int(line['end'])
                try:
                    gene_id = attribs['gene_id']
                except:
                    continue

                gbkey = attribs['gbkey']
                transcript_biotype = attribs['transcript_biotype']
                transcript_id = attribs['transcript_id']
                if transcript_id in skip_transcripts:
                    continue

                transcripts[transcript_id] = {"seqname":contig, "start":start, "end":end, "strand":strand, "gbkey":gbkey, "transcript_biotype":transcript_biotype}
                transcript2gene[transcript_id] = gene_id
                gene2transcripts[gene_id].add(transcript_id)
     
    if only_longest_transcript:
        transcript_lengths = defaultdict(int)
        for exon in exons:
            transcript_lengths[exon[1]] += exons[exon]['end'] - exons[exon]['start']

        keep_transcripts = dict()
        keep_exons = dict()
        keep_exon2transcript = dict()
        keep_transcript2gene = dict()
        keep_transcript2exon = defaultdict(set)
        keep_skip_transcripts = set()

        for gene in gene2transcripts:
            this_transcripts = gene2transcripts[gene]
            this_transcript_lengths = [(transcript, transcript_lengths[transcript]) for transcript in this_transcripts]
            longest_transcript = max(this_transcript_lengths, key=lambda x: x[1])[0]
            keep_transcripts[longest_transcript] = dict(transcripts[longest_transcript])
            for exon in transcript2exon[longest_transcript]:
                keep_exons[exon] = dict(exons[exon])
                keep_exon2transcript[exon] = longest_transcript
                keep_transcript2exon[longest_transcript].add(exon)
                keep_transcript2gene[longest_transcript] = gene
        
        transcripts = keep_transcripts
        exons = keep_exons
        exon2transcript = keep_exon2transcript
        transcript2gene = keep_transcript2gene
        transcript2exon = keep_transcript2exon
        skip_transcripts = keep_skip_transcripts

    return {
        'transcripts': transcripts,
        'exons': exons,
        'exon2transcript': exon2transcript,
        'transcript2gene': transcript2gene,
        'transcript2exon': transcript2exon
    }
    
def extract_default_transcript_sequences(transcript_info, fasta_records, output_file):

    for transcript_id in transcript_info['transcripts']:
        gene_id = transcript_info['transcript2gene'][transcript_id]
        this_exons = list(sorted(transcript_info['transcript2exon'][transcript_id], key=lambda x: x[-1]))

        seqname = None
        exon_qc_failed = False
        if len(this_exons) > 1:
            for i in range(1, len(this_exons)):
                this_exon = this_exons[i]
                prev_exon = this_exons[i-1]
                this_coords = transcript_info['exons'][this_exon]
                prev_coords = transcript_info['exons'][prev_exon]
                if this_coords['strand'] != prev_coords['strand']:
                    exon_qc_failed = True
                if this_coords['strand'] == '+' and this_coords['start'] < prev_coords['start']:
                    exon_qc_failed = True
                if this_coords['strand'] == '-' and this_coords['start'] > prev_coords['start']:
                    exon_qc_failed = True
                if this_coords['seqname'] != prev_coords['seqname']:
                    exon_qc_failed = True
    
        if exon_qc_failed:
            continue
    
        transcript_seq = ''
        for exon in this_exons:
            coords = transcript_info['exons'][exon]
            if seqname is None:
                seqname = coords['seqname']
            exon_seq = str(fasta_records[coords['seqname']][coords['start']:coords['end']])
            if coords['strand'] == '-':
                exon_seq = reverse_sequence(complement_sequence(exon_seq))
            transcript_seq += exon_seq
            
        print(f'>{seqname}|{gene_id}|{transcript_id}\n{transcript_seq}', file=output_file)

def extract_stitched_transcript_sequences(transcript_info, fasta_records, output_file, stitch_token='@', promoter_size=1024, intron_window=32, overlap=False):

    for transcript_id in transcript_info['transcripts']:
        gene_id = transcript_info['transcript2gene'][transcript_id]
        this_exons = list(sorted(transcript_info['transcript2exon'][transcript_id], key=lambda x: x[-1]))

        exon_qc_failed = False
        if len(this_exons) > 1:
            for i in range(1, len(this_exons)):
                this_exon = this_exons[i]
                prev_exon = this_exons[i-1]
                this_coords = transcript_info['exons'][this_exon]
                prev_coords = transcript_info['exons'][prev_exon]
                if this_coords['strand'] != prev_coords['strand']:
                    exon_qc_failed = True
                if this_coords['strand'] == '+' and this_coords['start'] < prev_coords['start']:
                    exon_qc_failed = True
                if this_coords['strand'] == '-' and this_coords['start'] > prev_coords['start']:
                    exon_qc_failed = True
                if this_coords['seqname'] != prev_coords['seqname']:
                    exon_qc_failed = True
    
        if exon_qc_failed:
            continue

        transcript_seq = ""
        seqname = None
        for i in range(len(this_exons)):
            # Previous Exon
            prev_exon = this_exons[i-1] if i > 0 else None
            prev_coords = transcript_info['exons'].get(prev_exon, None)
            # Current Exon
            cur_exon = this_exons[i]
            cur_coords = transcript_info['exons'].get(cur_exon, None)
            exon_number = cur_exon[-1]
            if seqname is None:
                seqname = cur_coords['seqname']
            # Next Exon
            next_exon = this_exons[i+1] if i < len(this_exons)-1 else None
            next_coords = transcript_info['exons'].get(next_exon, None)
            # Extract the stitched spliced sequence without overlapping intron windows.
            intron_window_left = min(intron_window, math.floor(abs(cur_coords['start'] - prev_coords['end']) / 2)) if not overlap and prev_coords is not None else intron_window
            intron_window_right = min(intron_window, math.ceil(abs(next_coords['start'] - cur_coords['end']) / 2)) if not overlap and next_coords is not None else intron_window
            if cur_coords['strand'] == '+' and exon_number == 1:
                exon_start = cur_coords['start'] - promoter_size
                exon_end = cur_coords['end'] + intron_window_right
            elif cur_coords['strand'] == '-' and exon_number == 1:
                exon_start = cur_coords['start'] - intron_window_left
                exon_end = cur_coords['end'] + promoter_size
            else:
                exon_start = cur_coords['start'] - intron_window_left
                exon_end = cur_coords['end'] + intron_window_right
            exon_seq = str(fasta_records[cur_coords['seqname']][exon_start:exon_end])
            if cur_coords['strand'] == '-':
                exon_seq = stitch_token + reverse_sequence(complement_sequence(exon_seq))
            transcript_seq += exon_seq
            
        if stitch_token and len(stitch_token) > 0:
            transcript_seq = transcript_seq[len(stitch_token):]
            
        print(f'>{seqname}|{gene_id}|{transcript_id}\n{transcript_seq}', file=output_file)

def run(args):

    with (
        open(args.output_path, "w") if args.output_path is not None else sys.stdout
    ) as output_file:

        if args.verbose:
            logging.info("Indexing FASTA file...")

        fasta_index = NvFaidx(args.fasta_path)

        if args.transcript_type == 'default':
            if args.verbose:
                logging.info("Extracting default transcripts...")
                if args.only_longest_transcript:
                    logging.info("Only extracting the longest transcript per gene.")
                else:
                    logging.info("Extracting all transcripts regardless of length.")
        
        elif args.transcript_type == 'stitched':
            if args.verbose:
                logging.info("Extracting stitched transcripts...")
                if args.only_longest_transcript:
                    logging.info("Only extracting the longest transcript per gene.")
                else:
                    logging.info("Extracting all transcripts regardless of length.")

        transcript_info = extract_transcript_exons(args.gtf_path, args.only_longest_transcript)

        if args.transcript_type == 'default':
            extract_default_transcript_sequences(transcript_info, fasta_index, output_file)
        elif args.transcript_type == 'stitched':
            extract_stitched_transcript_sequences(
                transcript_info,
                fasta_index,
                output_file,
                promoter_size=args.stitched_promoter,
                intron_window=args.stitched_intron,
                overlap=args.stitched_overlap
            )

def parse_args():
    """Parse command line arguments for splicing transcripts."""
    ap = argparse.ArgumentParser(description="Extract spliced transcripts from a FASTA and GTF.")
    ap.add_argument("--fasta-path", type=str, required=True, help="Path to FASTA file to extract transcripts from.")
    ap.add_argument("--gtf-path", type=str, required=True, help="Path to gene transfer format (GTF) file associated with the FASTA.")
    ap.add_argument("--output-path", type=str, default=None, help="Path to output FASTA file.")
    ap.add_argument("--transcript-type", type=str, default="default", choices=['default','stitched'],
                    help="Type of transcript to extract from the GTF and FASTA files for splicing. 'Stitched' transcripts include 1024 bp of sequence from the promoter and 32 bp around each exon.")
    ap.add_argument("--stitched-promoter", type=int, default=1024, help="Number of bp to include in the promoter region when --transcript-type=stitched is used. Defaults to 1024.")
    ap.add_argument("--stitched-intron", type=int, default=32, help="Number of bp to include from neighboring introns when --transcript-type=stitched is used. Defaults to 32.")
    ap.add_argument("--stitched-overlap", action='store_true',
                    help="Allow overlap of neighboring intron windows when --transcript-type=stitched is used. Defaults to False, i.e. prevents overlap by shortening the intron windows for a contiguous splice.")
    ap.add_argument("--only-longest-transcript", action='store_true', help="Only extract the longest transcript per gene.")
    ap.add_argument("-v", "--verbose", action='store_true', help="Turn on verbose log messages.")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.verbose:
        logging.info(args)
    run(args)

if __name__ == '__main__':
    main()