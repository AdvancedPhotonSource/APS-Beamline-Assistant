// @ts-expect-error no types for plotly.js-dist-min
import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)
export default Plot
