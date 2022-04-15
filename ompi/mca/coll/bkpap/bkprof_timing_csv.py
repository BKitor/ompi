#!/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/python
import argparse
import re
import pprint
import csv
import sys
pp = pprint.PrettyPrinter()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input_file", type=str, required=True, help="input file, should have BKPROF traces ", metavar="input_file")
parser.add_argument("-a", "--alg", type=str, default="ktreepipe", help="TODO: Implement algorithm selection: ['ktreepipe', 'ktree', 'rsa']", metavar="input_file")
args = parser.parse_args()

f_name = args.input_file

prof_inst_c = re.compile(
    r"^\[.+\]  BKPAP_PROFILE: (?P<time>\d+\.\d+) rank: (?P<rank>\d+) (?P<lbl>\w+)$")
	
lbl_dict = {
	"root": "ktree_pipeline_arrive",
	"points": [
		"finish_intra_reduce",
		"start_new_segment",
		"leave_new_seg_wait",
		"starting_new_sync_round",
		"synced_at_ss",
		"start_postbuf_reduce",
		"leave_postbuf_reduce",
		"reset_remote_ss",
		"got_parent_rank",
		"sent_parent_rank",
		"leave_main_loop",
		"leave_cleanup_wait",
		"final_cleanup_wait",
		"ktree_pipeline_leave",
	]
}


match_dict = {}
with open(f_name) as f:
	for i,l in enumerate(f):
		m = prof_inst_c.match(l)
		if m is None:
			continue
		m_time = float(m.group(1))
		m_rank = m.group(2)
		m_lbl = m.group(3)
		
		if m_rank not in match_dict:
			match_dict[m_rank] = [(0.0, "init")]
		
		prev_time = match_dict[m_rank][-1][0]
		match_dict[m_rank].append(( m_time - prev_time, m_lbl))

csv_writer = csv.writer(sys.stdout)
max_prof_len = max(map(len, match_dict.values()))

for i in range(max_prof_len):
	out_arr = []
	for rank in match_dict.values():
		if len(rank)> i:
			out_tuple = rank[i]
		else:
			out_tuple = (-1, "its_over")
		out_arr.append(f"{out_tuple[0]:2.5f} {out_tuple[1]}")
	csv_writer.writerow(out_arr)
