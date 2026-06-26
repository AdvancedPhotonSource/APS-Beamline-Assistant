import { useState } from 'react'
import { useChatStore } from '@/stores/chatStore'
import { useConfirmStore, nextConfirmId } from '@/stores/confirmStore'

interface MotorQuickAction {
  label: string
  prompt: string
}

const QUICK_ACTIONS: MotorQuickAction[] = [
  { label: 'List all motors', prompt: 'List all available motors and their current positions.' },
  { label: 'Show positions', prompt: 'Show the current positions of all motors.' },
  { label: 'Check status', prompt: 'Get the status of all motors including limits and velocities.' },
]

export function MotorDashboard() {
  const sendMessage = useChatStore((s) => s.sendMessage)
  const requestConfirm = useConfirmStore((s) => s.request)
  const [motorName, setMotorName] = useState('')
  const [targetPos, setTargetPos] = useState('')

  // Consequential motor actions are gated behind an explicit human confirm.
  // (The tool-layer safety in epics_motor_server still enforces limits server-side;
  // this is the visible, in-control front line.)
  const confirmMotor = (title: string, detail: string, run: () => void) => {
    requestConfirm({
      id: nextConfirmId(),
      title,
      detail,
      danger: true,
      safety: [
        'Soft/hard limits enforced server-side (HLM/LLM, HLS/LLS)',
        'Large-move guard: >50% travel needs explicit confirm',
      ],
      confirmLabel: 'Move motor',
      onConfirm: run,
    })
  }

  const handleMove = () => {
    if (!motorName.trim() || !targetPos.trim()) return
    const name = motorName, pos = targetPos
    confirmMotor(
      'Move motor?',
      `Move motor "${name}" to absolute position ${pos}.`,
      () => {
        sendMessage(`Move motor ${name} to position ${pos}.`)
        setMotorName('')
        setTargetPos('')
      },
    )
  }

  const handleJog = (dir: '-' | '+') => {
    if (!motorName.trim()) return
    const name = motorName
    confirmMotor(
      `Jog motor ${dir}?`,
      `Jog motor "${name}" in the ${dir === '-' ? 'negative' : 'positive'} direction.`,
      () => sendMessage(`Jog motor ${name} in the ${dir === '-' ? 'negative' : 'positive'} direction.`),
    )
  }

  return (
    <div className="p-3 space-y-4 overflow-y-auto h-full text-xs">
      {/* Quick actions */}
      <div>
        <div className="text-zinc-400 font-medium mb-2 uppercase tracking-wider text-[10px]">Quick Actions</div>
        <div className="space-y-1.5">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action.label}
              onClick={() => sendMessage(action.prompt)}
              className="w-full text-left px-3 py-2 rounded-lg border border-zinc-800 hover:border-zinc-600 hover:bg-zinc-800/50 text-zinc-300 transition-all"
            >
              {action.label}
            </button>
          ))}
        </div>
      </div>

      {/* Move motor form */}
      <div className="border-t border-zinc-800 pt-4">
        <div className="text-zinc-400 font-medium mb-2 uppercase tracking-wider text-[10px]">Move Motor</div>
        <div className="space-y-2">
          <input
            type="text"
            placeholder="Motor name (e.g. m1)"
            value={motorName}
            onChange={(e) => setMotorName(e.target.value)}
            className="w-full bg-zinc-800 text-zinc-200 rounded-md px-3 py-2 border border-zinc-700 focus:outline-none focus:border-blue-500 placeholder:text-zinc-600"
          />
          <input
            type="text"
            placeholder="Target position"
            value={targetPos}
            onChange={(e) => setTargetPos(e.target.value)}
            className="w-full bg-zinc-800 text-zinc-200 rounded-md px-3 py-2 border border-zinc-700 focus:outline-none focus:border-blue-500 placeholder:text-zinc-600"
          />
          <button
            onClick={handleMove}
            disabled={!motorName.trim() || !targetPos.trim()}
            className="w-full py-2 bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg transition-colors font-medium"
          >
            Move
          </button>
        </div>
      </div>

      {/* Jog controls */}
      <div className="border-t border-zinc-800 pt-4">
        <div className="text-zinc-400 font-medium mb-2 uppercase tracking-wider text-[10px]">Jog Motor</div>
        <div className="space-y-2">
          <input
            type="text"
            placeholder="Motor name (e.g. m1)"
            value={motorName}
            onChange={(e) => setMotorName(e.target.value)}
            className="w-full bg-zinc-800 text-zinc-200 rounded-md px-3 py-2 border border-zinc-700 focus:outline-none focus:border-blue-500 placeholder:text-zinc-600"
          />
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => handleJog('-')}
              disabled={!motorName.trim()}
              className="py-2 bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800/50 disabled:text-zinc-600 text-zinc-300 rounded-lg transition-colors flex items-center justify-center gap-1"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="15 18 9 12 15 6" />
              </svg>
              Jog -
            </button>
            <button
              onClick={() => handleJog('+')}
              disabled={!motorName.trim()}
              className="py-2 bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800/50 disabled:text-zinc-600 text-zinc-300 rounded-lg transition-colors flex items-center justify-center gap-1"
            >
              Jog +
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <div className="border-t border-zinc-800 pt-4">
        <button
          onClick={() => sendMessage('Stop all motors immediately.')}
          className="w-full py-2.5 bg-red-700 hover:bg-red-600 text-white rounded-lg transition-colors font-bold uppercase tracking-wide"
        >
          STOP ALL
        </button>
      </div>
    </div>
  )
}
