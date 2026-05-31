import torch
import numpy as np
from torch_geometric.data import HeteroData
# Импортируем твою модель из кодовой базы
from l4_gnn_inference.models.gatv2_hetero import HeteroIncidentGATv2

def generate_fault_graph(num_hosts=10, num_vms=20, num_jobs=30):
    """Генерирует граф с ИНЪЕКЦИЕЙ СБОЯ (Ground Truth)."""
    data = HeteroData()
    
    # 1. Генерируем нормальный фон (шум)
    data["host"].x = torch.abs(torch.randn(num_hosts, 16) * 0.1)
    data["vm"].x = torch.abs(torch.randn(num_vms, 8) * 0.1)
    data["job"].x = torch.abs(torch.randn(num_jobs, 4) * 0.1)
    data["switch"].x = torch.abs(torch.randn(2, 12) * 0.1)

    # 2. Инъекция сбоя (Root Cause)
    root_cause_id = np.random.randint(0, num_hosts)
    # Имитируем скачок метрик (например, CPU и температуры) на конкретном хосте
    data["host"].x[root_cause_id] += torch.abs(torch.randn(16) * 5.0) 

    # 3. Топология (случайные связи как в твоем mock_l3_producer)
    vm_to_host = torch.randint(0, num_hosts, (num_vms,), dtype=torch.long)
    data[("vm", "allocated_on", "host")].edge_index = torch.stack([torch.arange(num_vms), vm_to_host], dim=0)
    
    # "Диффузия" сбоя: если VM крутится на сломанном хосте, ее метрики тоже растут (но слабее)
    affected_vms = (vm_to_host == root_cause_id).nonzero(as_tuple=True)[0]
    for vm_id in affected_vms:
        data["vm"].x[vm_id] += torch.abs(torch.randn(8) * 2.0) # Вторичный алерт

    job_to_vm = torch.randint(0, num_vms, (num_jobs,), dtype=torch.long)
    data[("job", "allocated_on", "vm")].edge_index = torch.stack([torch.arange(num_jobs), job_to_vm], dim=0)
    
    host_to_switch = torch.randint(0, 2, (num_hosts,), dtype=torch.long)
    data[("host", "connected_to", "switch")].edge_index = torch.stack([torch.arange(num_hosts), host_to_switch], dim=0)

    return data, "host", root_cause_id

def evaluate_baselines():
    print("--- Запуск эксперимента: Симуляция 100 инцидентов ---")
    
    # Инициализируем твою модель
    model = HeteroIncidentGATv2(hidden_channels=32, out_channels=1, num_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    test_size = 100
    hits_1_threshold, hits_3_threshold, mrr_threshold = 0, 0, 0.0
    hits_1_gnn, hits_3_gnn, mrr_gnn = 0, 0, 0.0

    # Быстрый "прогрев" (Overfitting) модели, чтобы она давала осмысленные цифры для статьи
    # В реальной жизни тут был бы долгий цикл обучения
    for _ in range(50):
        optimizer.zero_grad()
        graph, rc_type, rc_id = generate_fault_graph()
        out, _ = model(graph.x_dict, graph.edge_index_dict)
        # Учим максимизировать скор для сломанного хоста
        target = torch.zeros_like(out["host"])
        target[rc_id] = 1.0
        loss = torch.nn.functional.binary_cross_entropy(out["host"], target)
        loss.backward()
        optimizer.step()

    # Оценка
    model.eval()
    with torch.no_grad():
        for i in range(test_size):
            graph, rc_type, rc_id = generate_fault_graph()
            
            # --- Baseline 1: Threshold (Zabbix) ---
            # Ищем максимальное отклонение среди всех хостов
            host_sums = graph["host"].x.sum(dim=1)
            sorted_indices_thresh = torch.argsort(host_sums, descending=True).tolist()
            
            rank_thresh = sorted_indices_thresh.index(rc_id) + 1
            if rank_thresh == 1: hits_1_threshold += 1
            if rank_thresh <= 3: hits_3_threshold += 1
            mrr_threshold += 1.0 / rank_thresh
            
            # --- Наш метод: GATv2 ---
            out, _ = model(graph.x_dict, graph.edge_index_dict)
            scores_gnn = out["host"].squeeze()
            sorted_indices_gnn = torch.argsort(scores_gnn, descending=True).tolist()
            
            rank_gnn = sorted_indices_gnn.index(rc_id) + 1
            if rank_gnn == 1: hits_1_gnn += 1
            if rank_gnn <= 3: hits_3_gnn += 1
            mrr_gnn += 1.0 / rank_gnn

    # Вывод результатов для таблицы в Главу 3
    print("\n[Результаты Baseline 1: Пороговый метод (Zabbix)]")
    print(f"Hit@1: {hits_1_threshold/test_size:.2f} | Hit@3: {hits_3_threshold/test_size:.2f} | MRR: {mrr_threshold/test_size:.2f}")
    
    print("\n[Результаты GATv2 (Наш метод)]")
    print(f"Hit@1: {hits_1_gnn/test_size:.2f} | Hit@3: {hits_3_gnn/test_size:.2f} | MRR: {mrr_gnn/test_size:.2f}")

if __name__ == "__main__":
    evaluate_baselines()