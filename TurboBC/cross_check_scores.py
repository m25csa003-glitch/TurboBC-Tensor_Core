#!/usr/bin/env python3
import sys
import os
import re
import json

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "bc_verification_results.log")
CPU_BASELINE_FILE = os.path.join(HERE, "cpu_baseline_scores.txt")
JSON_BASELINE_FILE = os.path.join(HERE, "cpu_baseline_scores.json")
AUDIT_REPORT_FILE = os.path.join(HERE, "gpu_vs_cpu_audit_report.txt")

if not os.path.exists(LOG_FILE):
    print(f"Error: {LOG_FILE} not found. Please run ./run_all_experiments.sh first.")
    sys.exit(1)

with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

blocks = content.split('[EXPERIMENT] Graph: ')
cpu_scores = {} # graph -> {node_id -> cpu_bc_score}
gpu_scores = {} # (graph, algo) -> {node_id -> gpu_bc_score}

for block in blocks[1:]:
    header = block.split('\n')[0]
    graph = header.split('.mtx')[0].strip()
    
    if 'scCSC' in header: algo = 'scCSC'
    elif 'vcCSC' in header or 'veCSC' in header or 'Warp' in header: algo = 'vcCSC'
    elif 'scCOOC' in header: algo = 'scCOOC'
    elif 'tcTF32' in header: algo = 'tcTF32'
    else: algo = 'Unknown'
    
    # Extract bc table: "bc   bc_hgpu \n0,0.000000,0.000000\n..."
    bc_section = re.search(r'bc\s+bc_hgpu\s*\n(.*?)(?=\n\n|\n---|\n===|\Z)', block, re.DOTALL)
    if not bc_section:
        continue
        
    lines = bc_section.group(1).strip().split('\n')
    if graph not in cpu_scores:
        cpu_scores[graph] = {}
        
    if (graph, algo) not in gpu_scores:
        gpu_scores[(graph, algo)] = {}
        
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            try:
                node_id = int(parts[0])
                cpu_val = float(parts[1])
                gpu_val = float(parts[2])
                
                # Store CPU baseline score
                if node_id not in cpu_scores[graph]:
                    cpu_scores[graph][node_id] = cpu_val
                
                gpu_scores[(graph, algo)][node_id] = gpu_val
            except ValueError:
                pass

# STEP 1: Write CPU Baseline Reference File
with open(CPU_BASELINE_FILE, 'w', encoding='utf-8') as f_txt:
    f_txt.write("========================================================================\n")
    f_txt.write("          CPU SEQUENTIAL BETWEENNESS CENTRALITY REFERENCE SCORES        \n")
    f_txt.write("          These baseline scores are stored for GPU cross-checking       \n")
    f_txt.write("========================================================================\n\n")
    for graph in sorted(cpu_scores.keys()):
        f_txt.write(f"--- GRAPH DATASET: {graph}.mtx (Total Verified Vertices: {len(cpu_scores[graph])}) ---\n")
        f_txt.write("Node_ID | CPU_BC_Score\n")
        for node in sorted(cpu_scores[graph].keys()):
            f_txt.write(f"{node:<7} | {cpu_scores[graph][node]:.6f}\n")
        f_txt.write("\n")

with open(JSON_BASELINE_FILE, 'w', encoding='utf-8') as f_json:
    json.dump(cpu_scores, f_json, indent=2)

# STEP 2: Perform Cross-Check Against Baseline & Write Audit Report
print("\n" + "="*96)
print("             🔍 CPU BASELINE STORAGE & GPU CROSS-CHECK AUDIT SUMMARY              ")
print("="*96)
print(f"📁 [Step 1] Stored CPU reference BC scores in : {CPU_BASELINE_FILE}")
print(f"🧪 [Step 2] Cross-checking GPU computed scores against stored CPU baseline...")
print("-" * 96)
print(f"{'Graph Name':<14} | {'GPU Algorithm':<13} | {'Total Nodes':<12} | {'Matched':<10} | {'Mismatched':<10} | {'Audit Status'}")
print("-" * 96)

audit_lines = [
    "========================================================================",
    "              GPU vs CPU CROSS-CHECK AUDIT VERIFICATION REPORT          ",
    "              Every GPU result checked against cpu_baseline_scores      ",
    "========================================================================\n"
]

for (graph, algo), g_data in sorted(gpu_scores.items(), key=lambda x: (x[0][0], x[0][1])):
    c_data = cpu_scores.get(graph, {})
    total_nodes = len(g_data)
    matched = 0
    mismatched = 0
    
    for node, gpu_val in g_data.items():
        cpu_val = c_data.get(node, 0.0)
        # Check tolerance for float32 IEEE-754 precision
        if abs(gpu_val - cpu_val) <= 1e-4 * max(1.0, abs(cpu_val)):
            matched += 1
        else:
            mismatched += 1
            
    status = "✅ PERFECT MATCH" if mismatched == 0 and total_nodes > 0 else "❌ FAILED"
    print(f"{graph:<14} | {algo:<13} | {total_nodes:<12} | {matched:<10} | {mismatched:<10} | {status}")
    
    audit_lines.append(f"Audit: Graph={graph:<14}, Algo={algo:<10} -> Checked={total_nodes:<5}, Matched={matched:<5}, Mismatched={mismatched} [{status}]")

with open(AUDIT_REPORT_FILE, 'w', encoding='utf-8') as f_audit:
    f_audit.write("\n".join(audit_lines))
    
print("-" * 96)
print(f"📄 Detailed Audit Report saved to: {AUDIT_REPORT_FILE}")
print("="*96 + "\n")
