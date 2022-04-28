#!/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/python
import argparse
import re
import pprint
import csv
import sys
pp = pprint.PrettyPrinter()

parser = argparse.ArgumentParser()
parser.add_argument("FILE", type=str, help="input file, should have BKPROF traces ", metavar="input_file")
parser.add_argument("-a", "--alg", type=str, default="ktreepipe", help="TODO: Implement algorithm selection: ['ktreepipe', 'ktree', 'rsa']", metavar="alg")
parser.add_argument("-n", "--num_proc", type=int, default=4, help="Number of processes", metavar="num_proc")
args = parser.parse_args()

f_name = args.FILE

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
	for l in f:
		m = prof_inst_c.match(l)
		if m is None:
			continue
		m_time = float(m.group('time'))
		m_rank = m.group('rank')
		m_lbl = m.group('lbl')
		
		if m_rank not in match_dict:
			match_dict[m_rank] = [(0.0, "init")]
		
		prev_time = match_dict[m_rank][-1][0]
		match_dict[m_rank].append(( m_time - prev_time, m_lbl))

max_prof_len = max(map(len, match_dict.values()))
csv_writer = csv.writer(sys.stdout)
csv_writer.writerow([f"rank {i}" for i in range(args.num_proc)])

for i in range(max_prof_len):
	out_arr = []
	for rank in range(args.num_proc):
		if len(match_dict[str(rank)])> i:
			out_tuple = match_dict[str(rank)][i]
		else:
			out_tuple = (-1, "its_over")
		out_arr.append(f"{out_tuple[0]:2.5f} {out_tuple[1]}")
	csv_writer.writerow(out_arr)
