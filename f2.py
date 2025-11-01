import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional, Any
import time
import random
import matplotlib.pyplot as plt

print("✅ Ambiente carregado com sucesso!")

# ========== CONFIGURAÇÃO COMPARTILHADA ==========
NUM_CLIENTS = 6
NUM_REGIONS = 2
NUM_ROUNDS = 10  # ✅ CONSTANTE
FAILURE_ROUNDS = [3, 4]  # 2 falhas (20%)

# ========== MODELO PYTORCH ==========
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

# ========== DADOS ==========
def load_mnist_data(client_id: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    clients_per_region = NUM_CLIENTS // NUM_REGIONS
    region_id = client_id // clients_per_region
    
    target_digits = [region_id * 2, region_id * 2 + 1]
    
    train_indices = [i for i, (_, label) in enumerate(train_dataset) 
                    if label in target_digits]
    train_subset = torch.utils.data.Subset(train_dataset, train_indices[:100])
    
    test_indices = [i for i, (_, label) in enumerate(test_dataset) 
                   if label in target_digits]
    test_subset = torch.utils.data.Subset(test_dataset, test_indices[:50])
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=10, shuffle=False)
    
    return train_loader, test_loader

# ========== CLIENTE BASE ==========
class BaseClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = SimpleNN()
        self.train_loader, self.test_loader = load_mnist_data(client_id)
        
    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            num_samples += data.size(0)
            
            if batch_idx >= 5:
                break
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        
        metrics = {
            'samples': num_samples,
            'loss': float(avg_loss)
        }
        
        return get_parameters(self.model), num_samples, metrics
    
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        
        return float(avg_loss), total, {"accuracy": float(accuracy)}

# ========== ESTRATÉGIA 1: TRADITIONAL CENTRALIZED FL ==========
class TraditionalFedAvg(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
        self.failed_rounds = []
        self.metrics_history = []
        self.successful_rounds = set()
        
    def configure_fit(self, server_round, parameters, client_manager):
        if server_round in FAILURE_ROUNDS:
            self.failed_rounds.append(server_round)
            print(f"❌ Servidor central falhou na rodada {server_round}")
            return []
        
        clients = client_manager.sample(num_clients=min(3, NUM_CLIENTS), min_num_clients=2)
        config = {'server_round': server_round}
        fit_ins = fl.common.FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        if server_round in self.failed_rounds:
            print(f"❌ Rodada {server_round}: Servidor central falhou - sem agregação")
            metrics = {
                'active_clients': 0,
                'failed_round': True,
                'successful_training': False
            }
            self.metrics_history.append({
                'round': server_round,
                'active_clients': 0,
                'failed': True,
                'successful_training': False
            })
            return None, metrics
        
        if len(results) < 2:
            print(f"⚠️ Rodada {server_round}: Clientes insuficientes ({len(results)})")
            metrics = {
                'active_clients': len(results),
                'failed_round': True,
                'successful_training': False
            }
            self.metrics_history.append({
                'round': server_round,
                'active_clients': len(results),
                'failed': True,
                'successful_training': False
            })
            return None, metrics
        
        aggregated_weights, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights:
            self.successful_rounds.add(server_round)
            metrics['active_clients'] = len(results)
            metrics['failed_round'] = False
            metrics['successful_training'] = True
            self.metrics_history.append({
                'round': server_round,
                'active_clients': len(results),
                'failed': False,
                'successful_training': True
            })
            print(f"✅ Rodada {server_round}: {len(results)} clients ativos")
        
        return aggregated_weights, metrics
    
    def get_resilience_metrics(self):
        successful_count = len([m for m in self.metrics_history if m.get('successful_training', False)])
        total_rounds = NUM_ROUNDS
        failure_rounds_completed = len([r for r in FAILURE_ROUNDS if r in self.successful_rounds])
        return successful_count, total_rounds, failure_rounds_completed

# ========== ESTRATÉGIA 2: HIERARCHICAL FL ==========
class HierarchicalFedAvg(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
        self.failed_regions = []
        self.metrics_history = []
        self.successful_rounds = set()
        
    def configure_fit(self, server_round, parameters, client_manager):
        current_failed_region = None
        if server_round in FAILURE_ROUNDS:
            failed_region = random.randint(0, NUM_REGIONS - 1)
            self.failed_regions.append((server_round, failed_region))
            current_failed_region = failed_region
            print(f"⚠️  Supernó {failed_region} falhou na rodada {server_round}")
        
        all_clients = list(client_manager.all().values())
        available_clients = []
        
        for client in all_clients:
            client_id = int(client.cid)
            client_region = client_id % NUM_REGIONS
            if current_failed_region is None or client_region != current_failed_region:
                available_clients.append(client)
        
        num_to_sample = min(3, len(available_clients))
        if num_to_sample < 2:
            print(f"❌ Rodada {server_round}: Região {current_failed_region} falhou - clientes insuficientes")
            return []
            
        clients = random.sample(available_clients, num_to_sample)
        config = {'server_round': server_round}
        fit_ins = fl.common.FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        current_failed_region = None
        for round_num, region in self.failed_regions:
            if round_num == server_round:
                current_failed_region = region
                break
        
        if not results or len(results) < 2:
            print(f"❌ Rodada {server_round}: Nenhum resultado válido ou clientes insuficientes")
            metrics = {
                'active_clients': len(results) if results else 0,
                'failed_round': True,
                'successful_training': False,
                'failed_region': current_failed_region is not None
            }
            self.metrics_history.append({
                'round': server_round,
                'active_clients': len(results) if results else 0,
                'failed': True,
                'successful_training': False
            })
            return None, metrics
        
        aggregated_weights, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights:
            self.successful_rounds.add(server_round)
            metrics['active_clients'] = len(results)
            metrics['failed_round'] = False
            metrics['successful_training'] = True
            metrics['failed_region'] = current_failed_region is not None
            
            self.metrics_history.append({
                'round': server_round,
                'active_clients': len(results),
                'failed': current_failed_region is not None,
                'successful_training': True
            })
            
            status = "com falha regional" if current_failed_region is not None else "normal"
            print(f"✅ Rodada {server_round}: {len(results)} clients ativos ({status})")
        
        return aggregated_weights, metrics
    
    def get_resilience_metrics(self):
        successful_count = len([m for m in self.metrics_history if m.get('successful_training', False)])
        total_rounds = NUM_ROUNDS
        failure_rounds_completed = len([r for r in FAILURE_ROUNDS if r in self.successful_rounds])
        return successful_count, total_rounds, failure_rounds_completed

# ========== ESTRATÉGIA 3: GEOFL WITH FAILOVER OTIMIZADO ==========
class OptimizedClientFailoverAgent:
    def __init__(self, client_id: int, all_regions: List[int]):
        self.client_id = client_id
        self.all_regions = all_regions
        self.current_region = client_id % NUM_REGIONS
        self.original_region = self.current_region
        self.region_scores: Dict[int, float] = {}
        self.last_update_round: Dict[int, int] = {}
        self.region_quality: Dict[int, float] = {r: 1.0 for r in all_regions}
        
    def evaluate_regions(self, current_round: int, failed_regions: List[int], model_weights=None) -> int:
        best_region = self.current_region
        best_score = -1
        
        for region in self.all_regions:
            if region in failed_regions:
                score = 0.0
            else:
                latency = random.uniform(10, 50)
                load = random.uniform(0.1, 0.8)
                freshness = self._get_checkpoint_freshness(region, current_round)
                model_quality = self.region_quality.get(region, 0.5)
                fairness = self._get_fairness_score(region)
                
                score = (0.2 * (1/latency)) + (0.2 * (1/load)) + (0.3 * freshness) + (0.3 * model_quality)
            
            self.region_scores[region] = score
            
            if score > best_score:
                best_score = score
                best_region = region
        
        current_score = self.region_scores.get(self.current_region, 0)
        
        if best_score > current_score + 0.05:
            self.current_region = best_region
            print(f"🔄 Client {self.client_id} migrou para região {best_region} (score: {best_score:.3f})")
            
        return self.current_region
    
    def update_region_quality(self, region: int, loss: float):
        quality = max(0.1, 1.0 - loss)
        self.region_quality[region] = 0.7 * self.region_quality.get(region, 0.5) + 0.3 * quality
    
    def _get_checkpoint_freshness(self, region: int, current_round: int) -> float:
        last_known = self.last_update_round.get(region, 0)
        staleness = current_round - last_known
        return max(0.1, 1 - (staleness / 5))
    
    def _get_fairness_score(self, region: int) -> float:
        region_clients = sum(1 for cid in range(NUM_CLIENTS) if cid % NUM_REGIONS == region)
        return 1.0 / (region_clients + 1)

class SuperNodeCoordinator:
    def __init__(self, region_id: int):
        self.region_id = region_id
        self.checkpoint = None
        self.processed_updates = set()
        
    def save_checkpoint(self, model_weights, round_id: int, client_ids: List[int]):
        self.checkpoint = {
            'round': round_id,
            'model_weights': model_weights,
            'client_ids': client_ids,
        }
        for cid in client_ids:
            self.processed_updates.add((cid, round_id))
    
    def is_duplicate(self, client_id: int, round_id: int) -> bool:
        return (client_id, round_id) in self.processed_updates

class OptimizedGeoFLFailoverClient(BaseClient):
    def __init__(self, client_id: int):
        super().__init__(client_id)
        self.cfa = OptimizedClientFailoverAgent(client_id, list(range(NUM_REGIONS)))
        self.supernodes = {region: SuperNodeCoordinator(region) for region in range(NUM_REGIONS)}
        
    def fit(self, parameters, config):
        current_round = int(config.get('server_round', 0))
        failed_regions_str = config.get('failed_regions', '[]')
        failed_regions = eval(failed_regions_str) if isinstance(failed_regions_str, str) else []
        
        target_region = self.cfa.evaluate_regions(current_round, failed_regions, parameters)
        target_supernode = self.supernodes[target_region]
        
        if target_supernode.is_duplicate(self.client_id, current_round) and random.random() > 0.7:
            return self.get_parameters({}), 0, {"duplicate": True}
        
        result = super().fit(parameters, config)
        
        if result[1] > 0:
            self.cfa.last_update_round[target_region] = current_round
            self.cfa.update_region_quality(target_region, result[2].get('loss', 1.0))
            
        weights, samples, metrics = result
        metrics['region'] = target_region
        metrics['region_changed'] = (target_region != self.cfa.original_region)
        
        return weights, samples, metrics

class OptimizedGeoFLWithFailover(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            min_fit_clients=2, 
            min_evaluate_clients=2, 
            min_available_clients=2,
            fraction_fit=0.6
        )
        self.failed_regions = []
        self.supernodes = {region: SuperNodeCoordinator(region) for region in range(NUM_REGIONS)}
        self.metrics_history = []
        self.successful_rounds = set()
        
    def configure_fit(self, server_round, parameters, client_manager):
        if server_round in FAILURE_ROUNDS:
            failed_region = random.randint(0, NUM_REGIONS - 1)
            self.failed_regions = [failed_region]
            print(f"⚠️  Supernó {failed_region} falhou na rodada {server_round}")
        else:
            self.failed_regions = []
        
        clients = client_manager.sample(num_clients=min(4, NUM_CLIENTS), min_num_clients=2)
        config = {
            'server_round': server_round,
            'failed_regions': str(self.failed_regions)
        }
        fit_ins = fl.common.FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        valid_results = []
        region_changes = 0
        
        for client, fit_res in results:
            metrics = fit_res.metrics or {}
            if not metrics.get('duplicate', False):
                valid_results.append((client, fit_res))
                if metrics.get('region_changed', False):
                    region_changes += 1
        
        if region_changes > 0:
            print(f"🌐 Rodada {server_round}: {region_changes} clientes mudaram de região")
        
        if not valid_results or len(valid_results) < 2:
            print(f"❌ Rodada {server_round}: Nenhum resultado válido ou clientes insuficientes ({len(valid_results)})")
            metrics = {
                'active_clients': len(valid_results),
                'active_regions': NUM_REGIONS - len(self.failed_regions),
                'failed_round': True,
                'successful_training': False
            }
            self.metrics_history.append({
                'round': server_round,
                'active_clients': len(valid_results),
                'active_regions': NUM_REGIONS - len(self.failed_regions),
                'failed': len(self.failed_regions) > 0,
                'successful_training': False
            })
            return None, metrics
        
        aggregated_weights, metrics = super().aggregate_fit(server_round, valid_results, failures)
        
        if aggregated_weights:
            self.successful_rounds.add(server_round)
            client_ids = [int(client.cid) for client, _ in valid_results]
            
            for region, supernode in self.supernodes.items():
                if region not in self.failed_regions:
                    supernode.save_checkpoint(aggregated_weights, server_round, client_ids)
            
            region_participation = {}
            for client, fit_res in valid_results:
                region_metrics = fit_res.metrics or {}
                region = region_metrics.get('region', -1)
                region_participation[region] = region_participation.get(region, 0) + 1
            
            metrics['active_clients'] = len(valid_results)
            metrics['active_regions'] = NUM_REGIONS - len(self.failed_regions)
            metrics['region_participation'] = region_participation
            metrics['failed_round'] = False
            metrics['successful_training'] = True
            
            self.metrics_history.append({
                'round': server_round,
                'active_clients': len(valid_results),
                'active_regions': NUM_REGIONS - len(self.failed_regions),
                'failed': len(self.failed_regions) > 0,
                'successful_training': True,
                'region_participation': region_participation
            })
            
            print(f"✅ Rodada {server_round}: {len(valid_results)} clients, Regiões ativas: {metrics['active_regions']}")
            if region_participation:
                print(f"   Participação: {region_participation}")
        
        return aggregated_weights, metrics
    
    def get_resilience_metrics(self):
        successful_count = len([m for m in self.metrics_history if m.get('successful_training', False)])
        total_rounds = NUM_ROUNDS
        failure_rounds_completed = len([r for r in FAILURE_ROUNDS if r in self.successful_rounds])
        return successful_count, total_rounds, failure_rounds_completed

# ========== SIMULAÇÃO E COMPARAÇÃO ==========
def run_strategy(strategy_name, strategy_class, client_class):
    print(f"\n{'='*50}")
    print(f"🏃 Executando: {strategy_name}")
    print(f"{'='*50}")
    
    def client_fn(cid: str):
        return client_class(int(cid))
    
    strategy_instance = strategy_class()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy_instance,
        client_resources={"num_cpus": 0.5},
    )
    
    return history, strategy_instance

def main():
    print("🚀 INICIANDO COMPARAÇÃO DE ESTRATÉGIAS - 2 FALHAS")
    print(f"📊 Configuração: {NUM_CLIENTS} clients, {NUM_REGIONS} regiões, {NUM_ROUNDS} rodadas")
    print(f"⚡ Falhas simuladas nas rodadas: {FAILURE_ROUNDS}")
    print(f"📈 Total de falhas: {len(FAILURE_ROUNDS)} ({(len(FAILURE_ROUNDS)/NUM_ROUNDS)*100:.1f}% das rodadas)")
    
    strategies = [
        ("Traditional Centralized FL", TraditionalFedAvg, BaseClient),
        ("Hierarchical FL", HierarchicalFedAvg, BaseClient), 
        ("GeoFL with Failover (Otimizado)", OptimizedGeoFLWithFailover, OptimizedGeoFLFailoverClient)
    ]
    
    results = {}
    strategy_instances = {}
    
    for strategy_name, strategy_class, client_class in strategies:
        history, strategy_instance = run_strategy(strategy_name, strategy_class, client_class)
        results[strategy_name] = history
        strategy_instances[strategy_name] = strategy_instance
    
    print(f"\n{'='*70}")
    print("📈 RESUMO COMPARATIVO - 2 FALHAS (20%)")
    print(f"{'='*70}")
    
    for strategy_name, history in results.items():
        print(f"\n🔍 {strategy_name}:")
        
        strategy_instance = strategy_instances[strategy_name]
        
        successful_count, total_rounds, failure_rounds_completed = strategy_instance.get_resilience_metrics()
        resilience_rate = successful_count / total_rounds if total_rounds > 0 else 0
        
        if hasattr(history, 'losses_distributed') and history.losses_distributed:
            final_loss = "N/A"
            for i in range(len(history.losses_distributed)-1, -1, -1):
                round_num, loss = history.losses_distributed[i]
                if round_num in strategy_instance.successful_rounds:
                    final_loss = loss
                    break
            print(f"   • Loss Final: {final_loss}")
        
        if hasattr(history, 'metrics_distributed') and 'accuracy' in history.metrics_distributed:
            final_accuracy = "N/A"
            for i in range(len(history.metrics_distributed['accuracy'])-1, -1, -1):
                round_num, accuracy = history.metrics_distributed['accuracy'][i]
                if round_num in strategy_instance.successful_rounds:
                    final_accuracy = accuracy
                    break
            print(f"   • Acurácia Final: {final_accuracy}")
        
        print(f"   • Taxa de Resiliência: {resilience_rate:.1%} ({successful_count}/{total_rounds} rodadas)")
        print(f"   • Rodadas com Falha Concluídas: {failure_rounds_completed}/{len(FAILURE_ROUNDS)}")
        
        successful_list = sorted(strategy_instance.successful_rounds)
        failed_in_failure_rounds = [r for r in FAILURE_ROUNDS if r not in strategy_instance.successful_rounds]
        print(f"   • Rodadas bem-sucedidas: {successful_list}")
        if failed_in_failure_rounds:
            print(f"   • Rodadas de falha NÃO recuperadas: {failed_in_failure_rounds}")

if __name__ == "__main__":
    main()
