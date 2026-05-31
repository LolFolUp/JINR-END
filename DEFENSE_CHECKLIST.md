# System Ready for Defense

## GNN Component
- Model: GATv2, v3.0.0, 311,878 params
- Hit@1: 78.3%  Hit@3: 82.6% overall (93.3% per-type avg)
- Per-fault: switch=1.00 ram=0.93 hdd=0.87
- Inference: ~14ms per graph

## RAG/LLM Component
- Backend: Mistral 7B via Ollama
- Actions: CHECK_METRICS, APPLY_QOS, ISOLATE_NODE, MIGRATE_JOB, NOTIFY_OPERATOR, SCHEDULE_MAINTENANCE, RESTART_SERVICE
- Firewall: Semantic validation 100%
- TTR: 6.4s total (6.4s Mistral, <2ms pipeline)

## Visualization
- Animation: 15 sec, 3 scenarios (congestion -> hdd -> ram)
- Graph: 46 nodes, real topology, propagation arrows
- Metrics: Hit@k report, comparison chart

## Tests
- GNN: 12 tests
- RAG: 14 tests
- Integration: 1 full pipeline test
- Total: 26/26 PASSED

## Live Demo
Command: `python -m remediation.run`
Expected: Root cause -> Mistral actions -> Firewall PASSED -> Report

## Files to Show Commission
1. `D:\Vlad\JINR-rag-archive\legacy-2026-05-31\artifacts\visualizations\cluster_animation.mp4` (15 sec demo, archived)
2. `artifacts/visualizations/real_inference_combined.png` (graph + scores)
3. `artifacts/remediation_report.json` (live output)
4. Terminal: `python -m remediation.run` (live demo)

## Key Talking Points
1. Problem: Cascading faults in HPC — one failure triggers avalanche of symptoms
2. Solution: GATv2 graph neural network finds root cause among victims
3. Extension: RAG + Mistral generates specific remediation steps
4. Safety: Semantic firewall blocks dangerous commands
5. Performance: ~500ms warm inference on RTX 3080
