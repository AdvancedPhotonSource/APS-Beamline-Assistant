import Plot from '@/lib/plotly'

interface LineoutPlotProps {
  data: {
    x: number[]
    y: number[]
    xLabel?: string
    yLabel?: string
    title?: string
  }
}

export function LineoutPlot({ data }: LineoutPlotProps) {
  return (
    <div className="w-full h-full min-h-[300px]">
      <Plot
        data={[
          {
            x: data.x,
            y: data.y,
            type: 'scatter',
            mode: 'lines',
            name: data.yLabel ?? 'Intensity',
            line: { color: '#3b82f6', width: 1.5 },
          },
        ]}
        layout={{
          title: { text: data.title ?? 'Diffraction Pattern' },
          paper_bgcolor: 'transparent',
          plot_bgcolor: '#18181b',
          font: { color: '#a1a1aa', family: 'Inter, system-ui, sans-serif', size: 12 },
          margin: { t: 40, r: 20, b: 50, l: 60 },
          xaxis: {
            title: { text: data.xLabel ?? '2theta (degrees)' },
            gridcolor: '#27272a',
            zerolinecolor: '#3f3f46',
          },
          yaxis: {
            title: { text: data.yLabel ?? 'Intensity' },
            gridcolor: '#27272a',
            zerolinecolor: '#3f3f46',
          },
          autosize: true,
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'] as Plotly.ModeBarDefaultButtons[],
          displaylogo: false,
        }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
