// @ts-expect-error no types for plotly.js-dist-min
import Plotly from 'plotly.js-dist-min'
import factoryModule from 'react-plotly.js/factory'

// Handle CJS/ESM interop — factory.js exports default as a property
const createPlotlyComponent = (factoryModule as unknown as { default: typeof factoryModule }).default ?? factoryModule
const Plot = createPlotlyComponent(Plotly)
export default Plot
