export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  timestamp: number
  toolResults?: ToolResult[]
  artifacts?: VizArtifact[]
}

export interface ToolResult {
  tool: string
  status: 'success' | 'error' | 'completed' | 'failed' | 'warning'
  data: Record<string, unknown>
}

export type ArtifactType = 'plotly' | 'image' | 'table' | 'text' | 'diffraction'

export interface VizArtifact {
  id: string
  type: ArtifactType
  title: string
  data: unknown
  sourceMessageId: string
}

export interface WsOutgoing {
  type: 'chat' | 'change_model'
  message?: string
  model?: string
}

export interface WsIncoming {
  type: 'chat_response' | 'error' | 'analysis_progress' | 'model_changed' | 'tool_result'
  message?: string
  step?: string
  progress?: number
  model?: string
  tool?: string
  result?: string
}

export interface ServerStatus {
  status: string
  mcp_client_connected: boolean
  connected_servers: string[]
  active_connections: number
  available_models: Record<string, Record<string, string>>
}

// Phase 2 types

export interface BrowseEntry {
  name: string
  path: string
  is_dir: boolean
  size: number | null
  modified: number
  ext?: string
  is_diffraction?: boolean
}

export interface BrowseResult {
  path: string
  parent: string | null
  entries: BrowseEntry[]
}

export interface ImageStats {
  shape: number[]
  dtype: string
  min: number
  max: number
  mean: number
  std: number
}

export interface ImageLoadResult {
  success: boolean
  file_id: string
  filename: string
  stats: ImageStats
  preview: string
}

export interface CsvData {
  columns: string[]
  data: Record<string, number[]>
  rows: number
  file: string
}

export interface ViewerAdjustResult {
  success: boolean
  image: string
  colormap: string
  vmin: number
  vmax: number
  gamma: number
}
