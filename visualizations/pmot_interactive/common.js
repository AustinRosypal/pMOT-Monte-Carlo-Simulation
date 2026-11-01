(function () {
  "use strict";

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function format(value, digits = 5) {
    if (!Number.isFinite(value)) return "—";
    if (value === 0) return "0";
    const magnitude = Math.abs(value);
    if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(3);
    return Number(value.toPrecision(digits)).toString();
  }

  function queryString(parameters) {
    const query = new URLSearchParams();
    Object.entries(parameters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") query.set(key, value);
    });
    return query.toString();
  }

  async function api(path, parameters = {}) {
    const response = await fetch(`${path}?${queryString(parameters)}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  function setStatus(element, message, error = false) {
    element.textContent = message;
    element.classList.toggle("error", error);
  }

  function debounce(callback, delay = 180) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => callback(...args), delay);
    };
  }

  function setupCanvas(canvas, requestedHeight) {
    const width = Math.max(320, canvas.parentElement.clientWidth);
    const height = requestedHeight || Math.max(320, parseFloat(getComputedStyle(canvas).height));
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { context, width, height };
  }

  function extent(values) {
    let minimum = Infinity;
    let maximum = -Infinity;
    values.forEach((value) => {
      if (Number.isFinite(value)) {
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
    });
    return [minimum, maximum];
  }

  function quantile(sorted, probability) {
    if (!sorted.length) return NaN;
    const index = Math.max(0, Math.min(sorted.length - 1, probability * (sorted.length - 1)));
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const fraction = index - lower;
    return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
  }

  function ticks(minimum, maximum, count = 6) {
    if (minimum === maximum) return [minimum];
    const values = [];
    for (let index = 0; index < count; index += 1) {
      values.push(minimum + (index * (maximum - minimum)) / (count - 1));
    }
    return values;
  }

  function drawLineChart(canvas, configuration, hoverIndex = null) {
    const { context, width, height } = setupCanvas(canvas, configuration.height || 360);
    const margin = { left: 76, right: 26, top: 24, bottom: 58 };
    const plot = {
      left: margin.left,
      right: width - margin.right,
      top: margin.top,
      bottom: height - margin.bottom,
    };
    const xValues = configuration.x;
    const visibleSeries = configuration.series.filter((series) => series.visible !== false);
    let [xMin, xMax] = configuration.xDomain || extent(xValues);
    const allY = [];
    visibleSeries.forEach((series) => {
      series.values.forEach((value, index) => {
        if (xValues[index] >= xMin && xValues[index] <= xMax && Number.isFinite(value)) {
          allY.push(value);
        }
      });
    });
    let yMin;
    let yMax;
    if (configuration.yDomain) {
      [yMin, yMax] = configuration.yDomain;
    } else {
      const sorted = allY.slice().sort((a, b) => a - b);
      const retained = Math.max(0.5, Math.min(1, configuration.robustFraction || 1));
      const tail = (1 - retained) / 2;
      yMin = quantile(sorted, tail);
      yMax = quantile(sorted, 1 - tail);
      if (configuration.includeZero !== false) {
        yMin = Math.min(yMin, 0);
        yMax = Math.max(yMax, 0);
      }
    }
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) [yMin, yMax] = [-1, 1];
    if (yMin === yMax) {
      const padding = Math.abs(yMin || 1) * 0.1;
      yMin -= padding;
      yMax += padding;
    }
    const yPadding = 0.06 * (yMax - yMin);
    yMin -= yPadding;
    yMax += yPadding;
    const xToPixel = (value) => plot.left + ((value - xMin) / (xMax - xMin)) * (plot.right - plot.left);
    const yToPixel = (value) => plot.bottom - ((value - yMin) / (yMax - yMin)) * (plot.bottom - plot.top);
    const xFromPixel = (pixel) => xMin + ((pixel - plot.left) / (plot.right - plot.left)) * (xMax - xMin);

    context.clearRect(0, 0, width, height);
    context.fillStyle = css("--surface");
    context.fillRect(0, 0, width, height);

    (configuration.bands || []).forEach((band) => {
      const left = Math.max(plot.left, xToPixel(band.x0));
      const right = Math.min(plot.right, xToPixel(band.x1));
      if (right > left) {
        context.fillStyle = band.color || css("--wall");
        context.fillRect(left, plot.top, right - left, plot.bottom - plot.top);
      }
    });

    context.font = "12px system-ui, sans-serif";
    context.lineWidth = 1;
    ticks(yMin, yMax, 6).forEach((tick) => {
      const pixel = yToPixel(tick);
      context.strokeStyle = css("--grid");
      context.beginPath();
      context.moveTo(plot.left, pixel);
      context.lineTo(plot.right, pixel);
      context.stroke();
      context.fillStyle = css("--muted");
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(format(tick, 4), plot.left - 9, pixel);
    });
    ticks(xMin, xMax, width < 600 ? 5 : 7).forEach((tick) => {
      const pixel = xToPixel(tick);
      context.strokeStyle = css("--grid");
      context.beginPath();
      context.moveTo(pixel, plot.top);
      context.lineTo(pixel, plot.bottom);
      context.stroke();
      context.fillStyle = css("--muted");
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(format(tick, 6), pixel, plot.bottom + 8);
    });
    if (xMin <= 0 && xMax >= 0) {
      context.strokeStyle = css("--muted");
      context.beginPath();
      context.moveTo(xToPixel(0), plot.top);
      context.lineTo(xToPixel(0), plot.bottom);
      context.stroke();
    }
    if (yMin <= 0 && yMax >= 0) {
      context.strokeStyle = css("--muted");
      context.beginPath();
      context.moveTo(plot.left, yToPixel(0));
      context.lineTo(plot.right, yToPixel(0));
      context.stroke();
    }

    (configuration.markers || []).forEach((marker) => {
      if (marker.x < xMin || marker.x > xMax) return;
      const pixel = xToPixel(marker.x);
      context.strokeStyle = marker.color || css("--series-4");
      context.setLineDash([5, 4]);
      context.beginPath();
      context.moveTo(pixel, plot.top);
      context.lineTo(pixel, plot.bottom);
      context.stroke();
      context.setLineDash([]);
    });

    visibleSeries.forEach((series) => {
      context.strokeStyle = series.color;
      context.lineWidth = series.width || 2;
      context.setLineDash(series.dash || []);
      context.beginPath();
      let started = false;
      series.values.forEach((value, index) => {
        const x = xValues[index];
        if (!Number.isFinite(value) || x < xMin || x > xMax || value < yMin || value > yMax) {
          started = false;
          return;
        }
        const px = xToPixel(x);
        const py = yToPixel(value);
        if (started) context.lineTo(px, py);
        else {
          context.moveTo(px, py);
          started = true;
        }
      });
      context.stroke();
      context.setLineDash([]);
    });

    if (hoverIndex !== null && hoverIndex >= 0 && hoverIndex < xValues.length) {
      const x = xValues[hoverIndex];
      if (x >= xMin && x <= xMax) {
        const pixel = xToPixel(x);
        context.strokeStyle = css("--fg");
        context.globalAlpha = 0.45;
        context.beginPath();
        context.moveTo(pixel, plot.top);
        context.lineTo(pixel, plot.bottom);
        context.stroke();
        context.globalAlpha = 1;
        visibleSeries.forEach((series) => {
          const value = series.values[hoverIndex];
          if (!Number.isFinite(value) || value < yMin || value > yMax) return;
          context.fillStyle = series.color;
          context.beginPath();
          context.arc(pixel, yToPixel(value), 3.5, 0, Math.PI * 2);
          context.fill();
        });
      }
    }

    context.strokeStyle = css("--border");
    context.strokeRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
    context.fillStyle = css("--fg");
    context.font = "13px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText(configuration.xLabel || "", (plot.left + plot.right) / 2, height - 4);
    context.save();
    context.translate(15, (plot.top + plot.bottom) / 2);
    context.rotate(-Math.PI / 2);
    context.fillText(configuration.yLabel || "", 0, 0);
    context.restore();

    const state = {
      kind: "line",
      configuration,
      plot,
      xMin,
      xMax,
      xToPixel,
      xFromPixel,
      redraw: (index = null) => drawLineChart(canvas, configuration, index),
    };
    canvas._vizState = state;
    bindInteractions(canvas);
    return state;
  }

  function parseHex(color) {
    const match = /^#([0-9a-f]{6})$/i.exec(color);
    if (!match) return [127, 127, 127];
    const value = parseInt(match[1], 16);
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
  }

  function mix(left, right, fraction) {
    const a = parseHex(left);
    const b = parseHex(right);
    const values = a.map((value, index) => Math.round(value + (b[index] - value) * fraction));
    return `rgb(${values.join(",")})`;
  }

  function heatColor(value, scale, diverging) {
    const surface = css("--surface");
    if (!diverging) {
      const fraction = Math.max(0, Math.min(1, (value - scale.min) / (scale.max - scale.min || 1)));
      return mix(surface, css("--series-1"), fraction);
    }
    const fraction = Math.min(1, Math.abs(value) / (scale.absMax || 1));
    return mix(surface, value < 0 ? css("--negative") : css("--positive"), fraction);
  }

  function drawHeatmap(canvas, configuration) {
    const requestedHeight = configuration.height || Math.max(420, Math.min(680, canvas.parentElement.clientWidth * 0.72));
    const { context, width, height } = setupCanvas(canvas, requestedHeight);
    const margin = { left: 76, right: 84, top: 22, bottom: 58 };
    const plot = { left: margin.left, right: width - margin.right, top: margin.top, bottom: height - margin.bottom };
    const x = configuration.x;
    const y = configuration.y;
    const values = configuration.values;
    const [xMin, xMax] = extent(x);
    const [yMin, yMax] = extent(y);
    const finite = values.flat().filter(Number.isFinite);
    const sortedAbs = finite.map(Math.abs).sort((a, b) => a - b);
    const robust = Math.max(0.5, Math.min(1, configuration.robustFraction || 0.99));
    const absMax = quantile(sortedAbs, robust) || 1;
    const [valueMin, valueMax] = extent(finite);
    const scale = { min: valueMin, max: valueMax, absMax };
    const xToPixel = (value) => plot.left + ((value - xMin) / (xMax - xMin)) * (plot.right - plot.left);
    const yToPixel = (value) => plot.bottom - ((value - yMin) / (yMax - yMin)) * (plot.bottom - plot.top);
    const xStep = (plot.right - plot.left) / Math.max(1, x.length - 1);
    const yStep = (plot.bottom - plot.top) / Math.max(1, y.length - 1);

    context.clearRect(0, 0, width, height);
    context.fillStyle = css("--surface");
    context.fillRect(0, 0, width, height);
    values.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        if (!Number.isFinite(value)) return;
        const px = xToPixel(x[columnIndex]) - xStep / 2;
        const py = yToPixel(y[rowIndex]) - yStep / 2;
        context.fillStyle = heatColor(value, scale, configuration.diverging !== false);
        context.fillRect(px, py, xStep + 1, yStep + 1);
      });
    });

    const vectors = configuration.vectors;
    if (vectors) {
      const stride = Math.max(1, configuration.vectorStride || Math.round(x.length / 13));
      context.strokeStyle = css("--fg");
      context.fillStyle = css("--fg");
      context.lineWidth = 1.15;
      for (let row = 0; row < y.length; row += stride) {
        for (let column = 0; column < x.length; column += stride) {
          if (vectors.defined && !vectors.defined[row][column]) continue;
          const u = vectors.u[row][column];
          const v = vectors.v[row][column];
          const out = vectors.out[row][column];
          const centerX = xToPixel(x[column]);
          const centerY = yToPixel(y[row]);
          const length = Math.min(xStep, yStep) * stride * 0.35;
          const endX = centerX + u * length;
          const endY = centerY - v * length;
          context.beginPath();
          context.moveTo(centerX, centerY);
          context.lineTo(endX, endY);
          context.stroke();
          const angle = Math.atan2(endY - centerY, endX - centerX);
          context.beginPath();
          context.moveTo(endX, endY);
          context.lineTo(endX - 5 * Math.cos(angle - 0.5), endY - 5 * Math.sin(angle - 0.5));
          context.lineTo(endX - 5 * Math.cos(angle + 0.5), endY - 5 * Math.sin(angle + 0.5));
          context.closePath();
          context.fill();
          if (Math.abs(out) > 0.35) {
            context.beginPath();
            context.arc(centerX, centerY, 3.5, 0, Math.PI * 2);
            context.stroke();
            if (out > 0) {
              context.beginPath();
              context.arc(centerX, centerY, 1.2, 0, Math.PI * 2);
              context.fill();
            } else {
              context.beginPath();
              context.moveTo(centerX - 2.2, centerY - 2.2);
              context.lineTo(centerX + 2.2, centerY + 2.2);
              context.moveTo(centerX + 2.2, centerY - 2.2);
              context.lineTo(centerX - 2.2, centerY + 2.2);
              context.stroke();
            }
          }
        }
      }
    }

    context.font = "12px system-ui, sans-serif";
    ticks(xMin, xMax, width < 600 ? 5 : 7).forEach((tick) => {
      const pixel = xToPixel(tick);
      context.fillStyle = css("--muted");
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(format(tick, 4), pixel, plot.bottom + 8);
    });
    ticks(yMin, yMax, 6).forEach((tick) => {
      const pixel = yToPixel(tick);
      context.fillStyle = css("--muted");
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(format(tick, 4), plot.left - 8, pixel);
    });
    context.strokeStyle = css("--border");
    context.strokeRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
    context.fillStyle = css("--fg");
    context.font = "13px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText(configuration.xLabel || "", (plot.left + plot.right) / 2, height - 4);
    context.save();
    context.translate(15, (plot.top + plot.bottom) / 2);
    context.rotate(-Math.PI / 2);
    context.fillText(configuration.yLabel || "", 0, 0);
    context.restore();

    const barLeft = plot.right + 22;
    const barWidth = 16;
    const steps = 100;
    for (let index = 0; index < steps; index += 1) {
      const fraction = index / (steps - 1);
      const value = configuration.diverging === false
        ? valueMin + fraction * (valueMax - valueMin)
        : -absMax + fraction * 2 * absMax;
      context.fillStyle = heatColor(value, scale, configuration.diverging !== false);
      const yPixel = plot.bottom - fraction * (plot.bottom - plot.top);
      context.fillRect(barLeft, yPixel, barWidth, (plot.bottom - plot.top) / steps + 1);
    }
    context.fillStyle = css("--muted");
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(
      format(configuration.diverging === false ? valueMax : absMax, 4),
      barLeft + barWidth + 5,
      plot.top,
    );
    context.fillText(
      format(configuration.diverging === false ? valueMin : -absMax, 4),
      barLeft + barWidth + 5,
      plot.bottom,
    );

    const state = {
      kind: "heatmap",
      configuration,
      plot,
      x,
      y,
      values,
      xMin,
      xMax,
      yMin,
      yMax,
      xToPixel,
      yToPixel,
      redraw: () => drawHeatmap(canvas, configuration),
    };
    canvas._vizState = state;
    bindInteractions(canvas);
    return state;
  }

  function nearestIndex(values, target) {
    let low = 0;
    let high = values.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (values[middle] < target) low = middle + 1;
      else high = middle;
    }
    if (low > 0 && Math.abs(values[low - 1] - target) < Math.abs(values[low] - target)) return low - 1;
    return low;
  }

  function bindInteractions(canvas) {
    if (canvas._vizBound) return;
    canvas._vizBound = true;
    const tooltip = canvas.parentElement.querySelector(".tooltip");
    canvas.addEventListener("mousemove", (event) => {
      const state = canvas._vizState;
      if (!state || !tooltip) return;
      const rectangle = canvas.getBoundingClientRect();
      const px = event.clientX - rectangle.left;
      const py = event.clientY - rectangle.top;
      if (px < state.plot.left || px > state.plot.right || py < state.plot.top || py > state.plot.bottom) {
        tooltip.style.display = "none";
        if (state.kind === "line") state.redraw(null);
        return;
      }
      if (state.kind === "line") {
        const xValue = state.xFromPixel(px);
        const index = nearestIndex(state.configuration.x, xValue);
        state.redraw(index);
        const rows = state.configuration.series
          .filter((series) => series.visible !== false)
          .map((series) => `${series.label}: <strong>${format(series.values[index], 6)}</strong>`)
          .join("<br>");
        tooltip.innerHTML = `${state.configuration.xLabel}: <strong>${format(state.configuration.x[index], 8)}</strong><br>${rows}`;
      } else {
        const xValue = state.xMin + ((px - state.plot.left) / (state.plot.right - state.plot.left)) * (state.xMax - state.xMin);
        const yValue = state.yMax - ((py - state.plot.top) / (state.plot.bottom - state.plot.top)) * (state.yMax - state.yMin);
        const column = nearestIndex(state.x, xValue);
        const row = nearestIndex(state.y, yValue);
        tooltip.innerHTML = `${state.configuration.xLabel}: <strong>${format(state.x[column], 6)}</strong><br>${state.configuration.yLabel}: <strong>${format(state.y[row], 6)}</strong><br>${state.configuration.valueLabel}: <strong>${format(state.values[row][column], 6)}</strong>`;
      }
      tooltip.style.display = "block";
      tooltip.style.left = `${Math.min(rectangle.width - tooltip.offsetWidth - 8, px + 14)}px`;
      tooltip.style.top = `${Math.max(6, py - tooltip.offsetHeight - 8)}px`;
    });
    canvas.addEventListener("mouseleave", () => {
      if (tooltip) tooltip.style.display = "none";
      if (canvas._vizState && canvas._vizState.kind === "line") canvas._vizState.redraw(null);
    });
  }

  function watchResize(callback) {
    const observer = new ResizeObserver(debounce(callback, 80));
    observer.observe(document.body);
    return observer;
  }

  window.PmotViz = {
    api,
    css,
    debounce,
    drawHeatmap,
    drawLineChart,
    format,
    queryString,
    setStatus,
    watchResize,
  };
})();
