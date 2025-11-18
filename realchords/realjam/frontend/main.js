/**
 * @fileoverview Launch point for genjam interface.
 *
 * @author Yusong Wu, Alex Scarlatos
 * Modified from https://github.com/lukewys/PianorollVis.js
 */

/** Kick off setup for all interface components. */
import { NoteVisual, DrawLoop, CONSTANTS } from './note_visualization.js';

window.onload = () => {
  SampleLibrary.setExt(".mp3");
  let div = document.querySelector('.loaded');
  let animationType = 'anticipation';
  let orientation = 'vertical';
  let numOctaves = -1;
  let lowestC = 1;
  let width = -1;
  let height = -1;
  let x = 0;
  let y = 0;
  let pianoAtBottom = true;
  const visual = new NoteVisual(
    div, animationType, orientation, numOctaves, lowestC, width, height, x, y,
    pianoAtBottom);
  window.addEventListener('resize', visual.onWindowResize.bind(visual));
  const drawLoop = new DrawLoop(CONSTANTS.REFRESH_RATE);
  drawLoop.addDrawFunctionFromVisual(visual);
  drawLoop.startDrawLoop();

  initializeMIDIReader(visual);
};