// API client for the Cartridge2 backend

export interface GameState {
  board: number[];
  current_player: number;
  winner: number;
  game_over: boolean;
  legal_moves: number[];
  message: string;
}

export interface MoveResponse extends GameState {
  bot_move: number | null;
}

export interface TrainingStats {
  epoch: number;
  loss: number;
  policy_loss: number;
  value_loss: number;
  games_played: number;
  learning_rate: number;
  timestamp: number;
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ModelInfo {
  loaded: boolean;
  path: string | null;
  file_modified: number | null;
  loaded_at: number | null;
  training_step: number | null;
  status: string;
}

const API_BASE = '';

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function getGameState(): Promise<GameState> {
  const res = await fetch(`${API_BASE}/game/state`);
  if (!res.ok) throw new Error('Failed to get game state');
  return res.json();
}

export async function newGame(first: 'player' | 'bot' = 'player'): Promise<GameState> {
  const res = await fetch(`${API_BASE}/game/new`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ first }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Failed to create new game');
  }
  return res.json();
}

export async function makeMove(position: number): Promise<MoveResponse> {
  const res = await fetch(`${API_BASE}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ position }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Move failed');
  }
  return res.json();
}

export async function getStats(): Promise<TrainingStats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to get stats');
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model`);
  if (!res.ok) throw new Error('Failed to get model info');
  return res.json();
}
