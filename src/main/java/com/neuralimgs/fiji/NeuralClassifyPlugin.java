package com.neuralimgs.fiji;

import java.awt.Choice;
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.Label;
import java.awt.Panel;
import java.awt.TextField;
import java.awt.Window;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.JDialog;
import javax.swing.JProgressBar;
import javax.swing.WindowConstants;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskException;

import ij.CompositeImage;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Roi;
import ij.measure.ResultsTable;
import ij.plugin.PlugIn;
import ij.plugin.frame.RoiManager;
import ij.process.LUT;
import ij.process.ShortProcessor;

/**
 * Runs the fixed-model single-cell classification pipeline (OPC / B3-Tub
 * ResNet-18 classifiers + RFP heuristic) on a CZI image and shows the results
 * in Fiji as ROIs + a results table.
 *
 * The image load, patch tiling, BaSiC illumination correction, CellPose
 * nucleus detection and PyTorch classification all happen in a Python
 * worker process managed by Appose (see {@code src/main/resources/scripts}):
 * there is no Java reimplementation of that pipeline, since BaSiC/CellPose/
 * PyTorch have no practical Java equivalent. This class is deliberately a
 * thin orchestration layer -- collect inputs, hand off to Python, render
 * what comes back.
 */
public class NeuralClassifyPlugin implements PlugIn {

	/**
	 * Cached across plugin invocations within the same Fiji session, so the
	 * Python environment (build once, ~minutes) and the loaded CellPose /
	 * PyTorch models (load once inside the worker, see
	 * classify_cell_image.py) are only paid for once per session.
	 */
	private static Environment environment;
	private static Service pythonService;

	/**
	 * Where the checkpoints are kept, and where {@link ModelBootstrap} puts them
	 * on first run. Under the user's home rather than beside the jar, because
	 * {@code Fiji.app/plugins/} is not reliably writable -- a macOS app bundle, a
	 * shared or read-only install -- and the download has to succeed without an
	 * administrator.
	 */
	private static final String DEFAULT_MODEL_DIR =
		new File(System.getProperty("user.home"), ".neural-imgs/models").getAbsolutePath();

	/**
	 * Bumped on every jar handed over, and shown in the dialog title + the Log.
	 *
	 * Fiji keeps a plugin's classes loaded until it is *restarted* -- "Refresh
	 * Menus" does not reload them -- so "I copied the new jar in" and "Fiji is
	 * running the new jar" are different claims. Without a visible stamp there
	 * is no way to tell a bug from a stale class.
	 */
	static final String BUILD_ID = "1.0.0";

	private static final int DEFAULT_PATCH_GRID = 4;

	/** Input modes, in dialog order. */
	private static final String[] INPUT_MODES = { "Single image", "Folder of images" };
	private static final int MODE_SINGLE = 0, MODE_FOLDER = 1;

	/** RFP rule names, in dialog order; index 0 is production. Must match the
	 *  literals RfpConfig.method accepts in neural_imgs.inference.rfp. */
	private static final String[] RFP_METHODS = { "neighbour", "legacy" };

	/** RFP defaults; must stay in step with RfpConfig in neural_imgs.inference.rfp,
	 *  which is what a run without these arguments would use, and with
	 *  RFP_NEIGHBOUR_THRESHOLD / RFP_SCORE_THRESHOLD in the notebook.
	 *
	 *  The two gates are separate fields on purpose: the rules are NOT on the
	 *  same scale (the background estimator differs), so one shared gate would
	 *  silently mean something far stricter or looser after switching rule. */
	private static final double DEFAULT_RFP_NEIGHBOUR_THRESHOLD = 16.5;
	private static final double DEFAULT_RFP_LEGACY_THRESHOLD = 5.0;
	private static final int DEFAULT_RFP_EXCLUDE_RADIUS = 60;

	/** Images allowed to be read + BaSiC-fitted ahead of the GPU lane. Each one
	 *  holds its whole patch array (~840 MB for a 16-patch OPC CZI), so this is
	 *  the batch's memory knob; 1 is enough to keep the GPU fed. */
	private static final int DEFAULT_PREFETCH = 1;

	/** CPU threads for per-patch cell extraction; 0 lets Python size the pool
	 *  from this process's core allocation. */
	private static final int DEFAULT_WORKERS = 0;

	/** CellPose detection defaults; must stay in step with FLOW_THRESHOLD /
	 *  CELLPROB_THRESHOLD in neural_imgs.inference.fixed_model_pipeline, which
	 *  is what a run without these arguments would use. */
	private static final double DEFAULT_CELLPROB_THRESHOLD = 0.1;
	private static final double DEFAULT_FLOW_THRESHOLD = 0.2;

	private static final double CELLPROB_MIN = -3.0, CELLPROB_MAX = 2.0;
	private static final double FLOW_MIN = 0.0, FLOW_MAX = 1.0;

	/** Field hints, shown in small grey italics under the field they describe.
	 *  The single-image and folder hints are greyed out together with their
	 *  field when the other input mode is selected (see setInputRowEnabled). */
	private static final String HINT_IMAGE =
		"Expects one Zeiss .czi image (5 channels, in order DAPI, OPC, RFP, B3Tub, BF).";
	private static final String HINT_FOLDER =
		"Expects a folder containing .czi images; every .czi directly inside it is run.";
	private static final String HINT_MODEL_DIR =
		"Downloaded automatically on first run; change only to use your own checkpoints.";
	private static final String HINT_OUT_DIR =
		"One subfolder per image is created inside it, named after the image.";

	/** Hint styling. The disabled colour is what marks a hint as "not this mode":
	 *  MultiLineLabel paints its own text and ignores setEnabled, so the greying
	 *  has to be done through the foreground colour. */
	private static final Font HINT_FONT = new Font("SansSerif", Font.ITALIC, 11);
	/** Group headings. Bold, so the required block at the top of the dialog is
	 *  visibly one group and not just the first few rows. */
	private static final Font SECTION_FONT = new Font("SansSerif", Font.BOLD, 12);
	private static final Color HINT_COLOR = new Color(0x55, 0x55, 0x55);
	private static final Color HINT_DISABLED_COLOR = new Color(0xAA, 0xAA, 0xAA);

	/** Shown by the dialog's "?" button. ImageJ renders a help string that starts
	 *  with {@code <html>} in its own window instead of opening a browser.
	 *
	 *  Sections are separated by rules and each one names the dialog fields it
	 *  explains, so the text can be read next to the dialog rather than in
	 *  dialog order. */
	private static final String DIALOG_HELP =
		"<html><body style='font-family:sans-serif;width:500px'>"
		+ "<h1 style='font-size:15pt;margin-bottom:2px'>Neural Image Classifier</h1>"
		+ "<p style='color:#555;margin-top:0'>Single-cell OPC / Neuron (B3-Tub) / "
		+ "Transduced (RFP) calls on Zeiss CZI images. Build " + BUILD_ID + ".</p>"

		+ "<hr>"
		+ "<h2>What one run does</h2>"
		+ "<p>The CZI is read and tiled into <b>Patch grid</b> &times; <b>Patch "
		+ "grid</b> patches, illumination-corrected (BaSiC), nuclei are detected "
		+ "on DAPI with CellPose, and every detected cell is classified by the two "
		+ "trained ResNet-18 models (OPC, B3-Tub) plus the RFP intensity rule. "
		+ "Results arrive as the <i>Neural Classifier Results</i> table, a CSV, an "
		+ "UpSet plot, and optionally ROIs and per-case example cells.</p>"
		+ "<p>The dialog reopens after each run with every field as you left it, so "
		+ "the next image is one field-change away. <b>Cancel</b> finishes.</p>"
		+ "<p>The dialog's top group -- input, model directory, output directory -- "
		+ "is everything a run <b>requires</b>. Every group below it is tuning, and "
		+ "its defaults are the ones the notebook uses.</p>"

		+ "<hr>"
		+ "<h2>First run: two one-time downloads</h2>"
		+ "<p>The very first run fetches the <b>Python environment</b> (a few "
		+ "hundred MB: PyTorch, CellPose, BaSiCPy) and the <b>trained models</b> "
		+ "(about 90 MB) and caches both. Expect several minutes and leave the "
		+ "progress window alone; every later run starts in seconds.</p>"
		+ "<p>The models land in the <i>Model directory</i> shown above, so there "
		+ "is nothing to copy by hand. Use <i>Plugins &gt; Neural Image Processor "
		+ "&gt; Install / Verify Models...</i> to do that download ahead of time, "
		+ "or to check what is installed. Point the field somewhere else only if "
		+ "you have your own retrained checkpoints -- an existing complete "
		+ "directory is never overwritten.</p>"

		+ "<hr>"
		+ "<h2>Input</h2>"
		+ "<p><b>Single image</b> classifies the one <code>.czi</code> you pick; "
		+ "the <i>Input folder</i> row is greyed out. <b>Folder of images</b> "
		+ "classifies every <code>.czi</code> directly inside the folder (not "
		+ "recursive, hidden files skipped, sorted by name); the <i>CZI image</i> "
		+ "row is greyed out. The greyed field keeps its text, so switching mode "
		+ "back does not mean retyping a path.</p>"
		+ "<p>Each image gets its own subfolder of the output directory, named "
		+ "after the image, holding its results table, quantification, figures and "
		+ "saved results. One image failing does not abort a folder run -- the "
		+ "failure is reported in the Log and the rest continue.</p>"

		+ "<hr>"
		+ "<h2>Tiling and nucleus detection</h2>"
		+ "<h3>Patch grid (NxN)</h3>"
		+ "<p>How many tiles the image is split into per side -- <b>4</b> means 16 "
		+ "patches. A tile count, not a pixel size: a whole number, at least 1. "
		+ "Tiling is what keeps one patch's pixels, its BaSiC correction and its "
		+ "CellPose call inside memory; cells are numbered patch by patch.</p>"
		+ "<h3>The two CellPose thresholds</h3>"
		+ "<p>They decide which nuclei are detected on the DAPI channel, and "
		+ "therefore which cells everything downstream classifies and counts.</p>"
		+ "<h3>Cell probability threshold</h3>"
		+ "<p>Does the pixel look like it belongs to a cell?<br>"
		+ "<b>Low threshold value: more cells detected</b><br>"
		+ "Values: [-3.0, -2.9, -2.8, &hellip;, 1.9, 2.0]<br>"
		+ "<i>Any decimal in that range is accepted, not just these steps.</i></p>"
		+ "<h3>Flow threshold</h3>"
		+ "<p>Does the proposed mask have flows consistent enough with what "
		+ "CellPose predicted?<br>"
		+ "<b>Low threshold value: less masks accepted (very strict)</b><br>"
		+ "Values: [0.0, 0.1, 0.2, &hellip;, 1.0]<br>"
		+ "<i>Any decimal in that range is accepted, not just these steps.</i></p>"
		+ "<p style='color:#555'>Changing either value invalidates saved results "
		+ "for that image, so the next run re-classifies from scratch.</p>"

		+ "<hr>"
		+ "<h2>RFP positivity</h2>"
		+ "<p>RFP has no trained classifier -- it is called from pixel intensity. "
		+ "Two rules are available and both are scored for every cell; the one "
		+ "selected here makes the call.</p>"
		+ "<h3>neighbour (production)</h3>"
		+ "<p>The cell's mean against the background of its crop <b>with pixels "
		+ "near other nuclei dropped</b>, so one bright neighbour can no longer "
		+ "set this cell's reference. Default gate <b>16.5</b>, exclusion radius "
		+ "<b>60 px</b>. Matches the notebook.</p>"
		+ "<h3>legacy</h3>"
		+ "<p>The original rule: mean over the dilated nucleus minus the crop's "
		+ "95th percentile. Default gate <b>5</b>. Kept for comparison.</p>"
		+ "<p style='color:#a00'><b>The two gates are not interchangeable.</b> The "
		+ "rules use different background estimators, so their scores are on "
		+ "different scales -- each has its own field above.</p>"

		+ "<hr>"
		+ "<h2>Performance -- what runs in parallel</h2>"
		+ "<p>Inside one image the patches are <b>pipelined, not serial</b>: "
		+ "CellPose runs on one patch at a time (one GPU, one model copy, so a "
		+ "second concurrent call would contend rather than add throughput) while "
		+ "that patch's cell extraction and RFP scoring run on a CPU thread pool, "
		+ "overlapping the next patch's CellPose call. <b>Extraction threads</b> "
		+ "sizes that pool; 0 sizes it from this process's core allocation.</p>"
		+ "<p>Across images in folder mode, reading the next CZI and fitting its "
		+ "BaSiC illumination model overlap the current image's GPU work. "
		+ "<b>Images prepared ahead</b> bounds how many may wait: each one holds "
		+ "its whole patch array (~840 MB for a 16-patch OPC CZI), so this is the "
		+ "run's memory knob. 1 is enough to keep the GPU lane fed.</p>"
		+ "<p style='color:#555'>Cell ids are assembled back in patch order, so "
		+ "the numbers are identical to a fully serial run.</p>"

		+ "<hr>"
		+ "<h2>Saved results</h2>"
		+ "<p><b>Load saved results if available</b> re-reads an image's previous "
		+ "results instead of classifying it again, and only when the image, patch "
		+ "grid, model directory, CellPose thresholds and RFP settings all match "
		+ "what was saved -- so a re-run over a mostly-done folder costs seconds. "
		+ "A cache hit never even reads the CZI.</p>"

		+ "<hr>"
		+ "<h2>Contact</h2>"
		+ "<p><b>Report a problem:</b> "
		+ "<a href='https://github.com/Amirhk-dev/neural-cell-classifier-fiji/issues'>"
		+ "github.com/Amirhk-dev/neural-cell-classifier-fiji/issues</a><br>"
		+ "Maintainer: Amirhossein Kardoost &mdash; "
		+ "<a href='mailto:kardoostamirhossein@gmail.com'>"
		+ "kardoostamirhossein@gmail.com</a></p>"
		+ "<p>Please include the build id above when reporting a problem, plus the "
		+ "Fiji Log contents of the failing run.</p>"

		+ "<hr>"
		+ "<h2>License</h2>"
		+ "<p><b>MIT License</b> &mdash; Copyright &copy; 2025 Amirhossein "
		+ "Kardoost.</p>"
		+ "<p style='color:#555'>Permission is hereby granted, free of charge, to "
		+ "any person obtaining a copy of this software and associated "
		+ "documentation files (the \"Software\"), to deal in the Software without "
		+ "restriction, including without limitation the rights to use, copy, "
		+ "modify, merge, publish, distribute, sublicense, and/or sell copies of "
		+ "the Software, and to permit persons to whom the Software is furnished to "
		+ "do so, subject to the following conditions:</p>"
		+ "<p style='color:#555'>The above copyright notice and this permission "
		+ "notice shall be included in all copies or substantial portions of the "
		+ "Software.</p>"
		+ "<p style='color:#555'>THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT "
		+ "WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO "
		+ "THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND "
		+ "NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE "
		+ "LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION "
		+ "OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION "
		+ "WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>"
		+ "<p style='color:#555'>CellPose, BaSiC, PyTorch and Fiji/ImageJ are "
		+ "used under their own licenses.</p>"
		+ "</body></html>";

	/**
	 * A {@link NonBlockingGenericDialog} that raises itself when shown.
	 *
	 * The prompt is re-shown from a background thread after a run that may have
	 * taken hours, by which point another window is focused; a non-modal dialog
	 * can then come back *behind* the main Fiji window and read as "the plugin
	 * closed". {@code GenericDialog.showDialog()} goes through
	 * {@code setVisible(true)}, so overriding it is the hook for that, with no
	 * timers or polling.
	 */
	private static final class FrontingDialog extends NonBlockingGenericDialog {
		FrontingDialog(final String title) {
			super(title);
		}

		@Override
		public void setVisible(final boolean visible) {
			super.setVisible(visible);
			if (visible) {
				toFront();
				requestFocus();
			}
		}
	}

	/** One classification run's settings, as collected from the dialog.
	 *
	 *  Both {@code imagePath} and {@code inputDir} are always carried; {@code
	 *  folderMode} says which one is live. Keeping the dead one rather than
	 *  blanking it is what lets the re-prompt after a run come back with both
	 *  fields still filled in, so switching mode does not mean retyping a path.
	 */
	private static final class ClassifyRequest {
		final boolean folderMode;
		final String imagePath;
		final String inputDir;
		final int patchGrid;
		final double cellprobThreshold;
		final double flowThreshold;
		final String rfpMethod;
		final double rfpNeighbourThreshold;
		final double rfpLegacyThreshold;
		final int rfpExcludeRadius;
		final int prefetch;
		final int workers;
		final String modelDir;
		final String outDir;
		final boolean generateUpset;
		final boolean showCaseMontage;
		final boolean addRois;
		final boolean reuseCache;

		ClassifyRequest(
			final boolean folderMode, final String imagePath, final String inputDir,
			final int patchGrid, final double cellprobThreshold, final double flowThreshold,
			final String rfpMethod, final double rfpNeighbourThreshold,
			final double rfpLegacyThreshold, final int rfpExcludeRadius,
			final int prefetch, final int workers,
			final String modelDir, final String outDir,
			final boolean generateUpset, final boolean showCaseMontage,
			final boolean addRois, final boolean reuseCache
		) {
			this.folderMode = folderMode;
			this.imagePath = imagePath;
			this.inputDir = inputDir;
			this.patchGrid = patchGrid;
			this.cellprobThreshold = cellprobThreshold;
			this.flowThreshold = flowThreshold;
			this.rfpMethod = rfpMethod;
			this.rfpNeighbourThreshold = rfpNeighbourThreshold;
			this.rfpLegacyThreshold = rfpLegacyThreshold;
			this.rfpExcludeRadius = rfpExcludeRadius;
			this.prefetch = prefetch;
			this.workers = workers;
			this.modelDir = modelDir;
			this.outDir = outDir;
			this.generateUpset = generateUpset;
			this.showCaseMontage = showCaseMontage;
			this.addRois = addRois;
			this.reuseCache = reuseCache;
		}

		/** What the run is over, for logs and error messages. */
		String describeInput() {
			return folderMode ? ("folder " + inputDir) : imagePath;
		}
	}

	/**
	 * Results of the last successful classification, kept so a later, separate
	 * "Show Cell..." menu invocation can look a cell up by id without
	 * re-running the pipeline. The Python-side worker independently caches the
	 * pixels needed to serve that lookup (see classify_cell_image.py).
	 */
	private static List<Map<String, Object>> lastCells;

	@Override
	public void run(final String arg) {
		try {
			if ("showCell".equals(arg)) {
				showCellDialog();
				return;
			}
			if ("installModels".equals(arg)) {
				installModelsDialog();
				return;
			}
			runClassifyLoop();
		}
		catch (final Exception e) {
			IJ.handleException(e);
		}
	}

	/**
	 * Prompt, run, prompt again -- until Cancel.
	 *
	 * A run takes hours and is usually one of several images, so closing the
	 * dialog for good after a single OK would mean walking back through the menu
	 * and retyping every field each time. The next prompt is pre-filled with what
	 * was just used, so a second image is one field-change away. A failed run
	 * returns to the prompt too, rather than ending the session.
	 */
	private void runClassifyLoop() {
		IJ.log("Neural Image Classifier: build " + BUILD_ID
			+ " -- the dialog reopens after each run; Cancel to finish.");
		ClassifyRequest previous = null;
		while (true) {
			final ClassifyRequest request = promptForRequest(previous);
			if (request == null) return;  // cancelled
			previous = request;
			try {
				classifyAndShow(request);
			}
			catch (final Exception e) {
				IJ.handleException(e);
			}
		}
	}

	/**
	 * Shows the settings dialog, re-prompting on invalid input; null if cancelled.
	 *
	 * Non-blocking, so the Log window and the Results table stay readable while
	 * it is up -- otherwise the results of the run just finished would be hidden
	 * behind the prompt asking about the next one.
	 */
	private ClassifyRequest promptForRequest(final ClassifyRequest previous) {
		while (true) {
			final ClassifyRequest request = showRequestDialog(previous);
			if (request == null) return null;
			final String problem = validate(request);
			if (problem == null) return request;
			IJ.error("Neural Image Classifier", problem);
		}
	}

	private ClassifyRequest showRequestDialog(final ClassifyRequest previous) {
		final NonBlockingGenericDialog gd =
			new FrontingDialog("Neural Image Classifier (" + BUILD_ID + ")");
		// Every field starts from what the last run used, so a follow-up image
		// only needs the path changed.
		//
		// Order: everything that has no usable default is asked first, so a run
		// can be started by filling the top of the dialog and leaving the rest
		// alone. Each group below the required block is a tuning section whose
		// defaults match the notebook.
		addSection(gd, "Required -- what to classify, with which models, where it goes");
		gd.addChoice("Input:", INPUT_MODES,
			INPUT_MODES[previous != null && previous.folderMode ? MODE_FOLDER : MODE_SINGLE]);
		// Each input row's three widgets are picked up as it is added -- the
		// caption from getLabel(), the text field from the string-field list, the
		// hint from getMessage() -- because those accessors all report the *most
		// recently added* component and there is no other handle to them. The
		// input-mode toggle then greys out whichever row is not in use.
		gd.addFileField("CZI image:", previous == null ? "" : previous.imagePath);
		final InputRow imageRow = InputRow.of(
			lastStringField(gd), gd.getLabel(), addHint(gd, HINT_IMAGE));
		gd.addDirectoryField("Input folder:", previous == null ? "" : previous.inputDir);
		final InputRow folderRow = InputRow.of(
			lastStringField(gd), gd.getLabel(), addHint(gd, HINT_FOLDER));
		gd.addDirectoryField("Model directory:",
			previous == null ? DEFAULT_MODEL_DIR : previous.modelDir);
		addHint(gd, HINT_MODEL_DIR);
		gd.addDirectoryField("Output directory:",
			previous == null ? System.getProperty("user.home") : previous.outDir);
		addHint(gd, HINT_OUT_DIR);

		addSection(gd, "Tiling and nucleus detection");
		// Patch grid is a tile count -> 0 decimals; the two thresholds are
		// genuinely continuous, so they get decimal places. The digit count only
		// formats the default shown -- ImageJ parses whatever is typed, at full
		// precision, either way, so validate() is what actually enforces
		// "whole number" and the allowed ranges.
		gd.addNumericField("Patch grid (NxN):",
			previous == null ? DEFAULT_PATCH_GRID : previous.patchGrid, 0);
		gd.addNumericField("Cell probability threshold:",
			previous == null ? DEFAULT_CELLPROB_THRESHOLD : previous.cellprobThreshold, 2);
		gd.addNumericField("Flow threshold:",
			previous == null ? DEFAULT_FLOW_THRESHOLD : previous.flowThreshold, 2);

		addSection(gd, "RFP positivity (no classifier -- an intensity rule)");
		gd.addChoice("RFP rule:", RFP_METHODS,
			previous == null ? RFP_METHODS[0] : previous.rfpMethod);
		gd.addNumericField("RFP gate, neighbour rule:",
			previous == null ? DEFAULT_RFP_NEIGHBOUR_THRESHOLD : previous.rfpNeighbourThreshold, 2);
		gd.addNumericField("RFP gate, legacy rule:",
			previous == null ? DEFAULT_RFP_LEGACY_THRESHOLD : previous.rfpLegacyThreshold, 2);
		gd.addNumericField("Neighbour exclusion radius (px):",
			previous == null ? DEFAULT_RFP_EXCLUDE_RADIUS : previous.rfpExcludeRadius, 0);

		addSection(gd, "Performance");
		gd.addNumericField("Images prepared ahead (memory):",
			previous == null ? DEFAULT_PREFETCH : previous.prefetch, 0);
		gd.addNumericField("Extraction threads (0 = auto):",
			previous == null ? DEFAULT_WORKERS : previous.workers, 0);

		addSection(gd, "Outputs");
		gd.addCheckbox("Generate UpSet plot (OPC / Neuron / Transduced)",
			previous == null || previous.generateUpset);
		gd.addCheckbox("Show one example cell per UpSet case (all channels + probabilities)",
			previous != null && previous.showCaseMontage);
		gd.addCheckbox("Add cell ROIs to the ROI Manager (opens it)",
			previous != null && previous.addRois);
		gd.addCheckbox("Load saved results if available (skip re-classifying)",
			previous == null || previous.reuseCache);
		// A "<html>..." help text is shown by ImageJ in its own window rather
		// than handed to a browser, so the plugin stays self-contained offline.
		// The button keeps ImageJ's default "Help" label.
		gd.addHelp(DIALOG_HELP);
		// Last, so the dialog is fully built before anything in it is greyed out.
		installInputModeToggle(gd, imageRow, folderRow);
		gd.showDialog();
		if (gd.wasCanceled()) return null;

		// Read in the exact order the fields were added -- GenericDialog's
		// getNext* calls are positional (one counter per field *type*), so moving
		// or inserting a field above must be mirrored here or every later field
		// of that type silently shifts by one.
		final boolean folderMode = gd.getNextChoiceIndex() == MODE_FOLDER;
		final String imagePath = gd.getNextString();
		final String inputDir = gd.getNextString();
		final String modelDir = gd.getNextString();
		final String outDir = gd.getNextString();
		// Read as a double, not (int): the cast alone would silently turn a
		// typed "4.7" into 4 rather than telling the user it was ignored.
		final double patchGridEntry = gd.getNextNumber();
		final double cellprobThreshold = gd.getNextNumber();
		final double flowThreshold = gd.getNextNumber();
		final String rfpMethod = gd.getNextChoice();
		final double rfpNeighbourThreshold = gd.getNextNumber();
		final double rfpLegacyThreshold = gd.getNextNumber();
		final double rfpExcludeRadiusEntry = gd.getNextNumber();
		final double prefetchEntry = gd.getNextNumber();
		final double workersEntry = gd.getNextNumber();
		final boolean generateUpset = gd.getNextBoolean();
		final boolean showCaseMontage = gd.getNextBoolean();
		final boolean addRois = gd.getNextBoolean();
		final boolean reuseCache = gd.getNextBoolean();

		// Out-of-range numbers are carried through to validate() rather than
		// rejected here, so a re-prompt can show them back with the complaint.
		return new ClassifyRequest(
			folderMode, imagePath, inputDir, (int) patchGridEntry,
			cellprobThreshold, flowThreshold,
			rfpMethod, rfpNeighbourThreshold, rfpLegacyThreshold,
			(int) rfpExcludeRadiusEntry, (int) prefetchEntry, (int) workersEntry,
			modelDir, outDir, generateUpset, showCaseMontage, addRois, reuseCache
		);
	}

	/**
	 * One file/directory row of the dialog: its text field, its Browse button,
	 * its caption and its hint, enabled and greyed out together.
	 *
	 * {@code addFileField} re-parents the text field into a {@link java.awt.Panel}
	 * together with the Browse button, so that panel -- not the text field -- is
	 * the row: disabling the text field alone would leave Browse live and able to
	 * overwrite it.
	 */
	private static final class InputRow {
		private final Component row;       // text field, or the panel holding it + Browse
		private final Label caption;
		private final Component hint;

		/** Null when there are no components to grey out, i.e. a headless or
		 *  macro-driven run, where the dialog builds no widgets. */
		static InputRow of(final TextField field, final Label caption, final Component hint) {
			return field == null ? null : new InputRow(field, caption, hint);
		}

		private InputRow(final TextField field, final Label caption, final Component hint) {
			final Container parent = field.getParent();
			// Not a panel means the field was added on its own, with no Browse
			// button beside it; then the field itself is the whole row.
			this.row = parent instanceof Panel ? parent : field;
			this.caption = caption;
			this.hint = hint;
		}

		void setEnabled(final boolean enabled) {
			if (row instanceof Panel) {
				for (final Component c : ((Panel) row).getComponents()) c.setEnabled(enabled);
			}
			row.setEnabled(enabled);
			if (caption != null) caption.setEnabled(enabled);
			if (hint != null) {
				// Greyed by colour, not by setEnabled: addMessage returns a plain
				// Label for a one-line hint but a MultiLineLabel otherwise, and
				// MultiLineLabel paints its own text and ignores setEnabled.
				hint.setForeground(enabled ? HINT_COLOR : HINT_DISABLED_COLOR);
				hint.repaint();
			}
		}
	}

	/** Adds a bold group heading, with space above it to separate it from the
	 *  group before. */
	private static void addSection(final GenericDialog gd, final String title) {
		gd.setInsets(12, 0, 2);   // applies to the next component only
		gd.addMessage(title, SECTION_FONT);
	}

	/**
	 * Adds a small grey italic hint line under the field just added, and returns
	 * it so its colour can be changed later.
	 *
	 * Indented to the field column so it reads as belonging to the field above
	 * rather than as a new section heading.
	 */
	private static Component addHint(final GenericDialog gd, final String text) {
		gd.setInsets(0, 20, 0);   // applies to the next component only
		gd.addMessage(text, HINT_FONT, HINT_COLOR);
		return gd.getMessage();
	}

	/** The text field of the field added last, or null in a headless/macro run. */
	private static TextField lastStringField(final GenericDialog gd) {
		final Vector<?> fields = gd.getStringFields();
		return fields == null || fields.isEmpty()
			? null : (TextField) fields.lastElement();
	}

	/**
	 * Greys out whichever input row the selected mode does not use, and keeps
	 * doing so as the mode is changed.
	 *
	 * Only one of "CZI image" and "Input folder" is ever read (see
	 * {@link #validate}), so leaving both live invites filling in the one that is
	 * ignored and reading the resulting "No valid ..." error as a bug. The
	 * disabled field keeps its text on purpose -- that is what lets switching
	 * mode back not mean retyping a path.
	 *
	 * Wired on the {@link Choice} itself rather than through a
	 * {@code DialogListener}, because a dialog listener has to read every field
	 * in order on each keystroke; this only needs the one selection.
	 */
	private static void installInputModeToggle(
		final GenericDialog gd, final InputRow imageRow, final InputRow folderRow
	) {
		final Vector<?> choices = gd.getChoices();
		if (choices == null || choices.isEmpty() || imageRow == null || folderRow == null) {
			return;   // headless or macro run: no widgets to grey out
		}
		final Choice mode = (Choice) choices.get(0);
		final Runnable sync = () -> {
			final boolean folderMode = mode.getSelectedIndex() == MODE_FOLDER;
			imageRow.setEnabled(!folderMode);
			folderRow.setEnabled(folderMode);
		};
		mode.addItemListener(event -> sync.run());
		sync.run();   // the pre-filled mode has to start out consistent too
	}

	/** First thing wrong with {@code request}, phrased for the user, or null. */
	private static String validate(final ClassifyRequest request) {
		if (request.folderMode) {
			if (request.inputDir == null || request.inputDir.isBlank()
				|| !Files.isDirectory(Paths.get(request.inputDir)))
			{
				return "No valid input folder selected.";
			}
		}
		else if (request.imagePath == null || request.imagePath.isBlank()
			|| !Files.exists(Paths.get(request.imagePath)))
		{
			return "No valid CZI image selected.";
		}
		if (request.outDir == null || request.outDir.isBlank()) {
			return "No output directory selected.";
		}
		if (request.rfpNeighbourThreshold < 0 || Double.isNaN(request.rfpNeighbourThreshold)
			|| request.rfpLegacyThreshold < 0 || Double.isNaN(request.rfpLegacyThreshold))
		{
			return "RFP gates must be numbers and cannot be negative.";
		}
		if (request.rfpExcludeRadius < 0) {
			return "Neighbour exclusion radius must be 0 or more pixels "
				+ "(0 reproduces the legacy background).";
		}
		if (request.prefetch < 1) {
			return "At least 1 image must be prepared ahead.";
		}
		if (request.workers < 0) {
			return "Extraction threads must be 0 (auto) or more.";
		}
		if (request.patchGrid < 1) {
			return "Patch grid must be a whole number of tiles per side, at least 1.";
		}
		if (Double.isNaN(request.cellprobThreshold)
			|| request.cellprobThreshold < CELLPROB_MIN
			|| request.cellprobThreshold > CELLPROB_MAX)
		{
			return "Cell probability threshold must be a number between " + CELLPROB_MIN
				+ " and " + CELLPROB_MAX + ".";
		}
		if (Double.isNaN(request.flowThreshold)
			|| request.flowThreshold < FLOW_MIN || request.flowThreshold > FLOW_MAX)
		{
			return "Flow threshold must be a number between " + FLOW_MIN
				+ " and " + FLOW_MAX + ".";
		}
		// A model directory that is empty or incomplete is NOT rejected here:
		// classifyAndShow() downloads the published bundle into it (see
		// ModelBootstrap), which is what makes a fresh install one jar and no
		// manual file copying. Only a directory this process could never write
		// is a dead end worth stopping for, since the download would fail
		// minutes later with a less obvious message.
		final Path modelDir = Paths.get(request.modelDir);
		if (!ModelBootstrap.isInstalled(modelDir) && !ModelBootstrap.canDownload()) {
			return "No models found in:\n" + request.modelDir + "\n\n"
				+ "This build has no model download configured, so the OPC + B3Tub "
				+ "checkpoint directories and thresholds.json have to be copied "
				+ "there by hand (see the project README).";
		}
		if (!ModelBootstrap.isInstalled(modelDir)) {
			final Path existing = nearestExistingAncestor(modelDir);
			if (existing != null && !Files.isWritable(existing)) {
				return "The models need to be downloaded into:\n" + request.modelDir
					+ "\n\nbut that location is not writable. Pick a model "
					+ "directory inside your home folder instead.";
			}
		}
		return null;
	}

	/**
	 * Runs (or, when {@code reuseCache} finds a matching entry, re-loads) the
	 * classification and renders it. Whether the numbers were recomputed or read
	 * back from disk is decided in Python -- see classify_cell_image.py and
	 * neural_imgs.inference.result_cache -- so the freshness rules live in one
	 * place; this side only reports what happened.
	 */
	private void classifyAndShow(final ClassifyRequest request)
		throws IOException, BuildException, InterruptedException, TaskException
	{
		// Before the Python environment, not after: building that environment is
		// the multi-minute step, and failing the download afterwards would make
		// the user wait through it only to be told the models are missing.
		installModels(Paths.get(request.modelDir));

		final Service python = getPythonService();

		final Map<String, Object> inputs = new HashMap<>();
		// Exactly one of these is non-empty; Python branches on input_dir rather
		// than on a separate mode flag, so a blank string is the signal and there
		// is no second thing that could disagree with it.
		inputs.put("image_path", request.folderMode ? "" : request.imagePath);
		inputs.put("input_dir", request.folderMode ? request.inputDir : "");
		inputs.put("patch_grid", request.patchGrid);
		inputs.put("cellprob_threshold", request.cellprobThreshold);
		inputs.put("flow_threshold", request.flowThreshold);
		inputs.put("rfp_method", request.rfpMethod);
		inputs.put("rfp_neighbour_threshold", request.rfpNeighbourThreshold);
		inputs.put("rfp_legacy_threshold", request.rfpLegacyThreshold);
		inputs.put("rfp_exclude_radius", request.rfpExcludeRadius);
		inputs.put("model_dir", request.modelDir);
		inputs.put("out_dir", request.outDir);
		inputs.put("generate_upset", request.generateUpset);
		inputs.put("show_case_montage", request.showCaseMontage);
		inputs.put("reuse_cache", request.reuseCache);
		inputs.put("prefetch", request.prefetch);
		inputs.put("n_workers", request.workers);

		final String script = readResource("/scripts/classify_cell_image.py");

		final Task task = python.task(script, inputs);
		task.listen(event -> {
			if (event.responseType == ResponseType.UPDATE) {
				IJ.log("Neural Image Classifier: " + event.message);
			}
		});
		IJ.log("Neural Image Classifier: running pipeline on " + request.describeInput()
			+ "  (cellprob " + request.cellprobThreshold
			+ ", flow " + request.flowThreshold
			+ ", RFP " + request.rfpMethod + " gate " + activeRfpGate(request) + ") ...");
		final long start = System.currentTimeMillis();
		task.start();
		task.waitFor();
		if (task.status != TaskStatus.COMPLETE) {
			throw new RuntimeException("Python task failed: " + task.error);
		}
		IJ.log("Neural Image Classifier: finished in "
			+ (System.currentTimeMillis() - start) / 1000. + " s");

		@SuppressWarnings("unchecked")
		final List<Map<String, Object>> cells =
			(List<Map<String, Object>>) task.outputs.get("cells");
		lastCells = cells;
		@SuppressWarnings("unchecked")
		final List<Map<String, Object>> images =
			(List<Map<String, Object>>) task.outputs.get("images");

		logBatch(task, images);
		renderResults(cells, request);
		showImageFigures(images, request);
	}

	/** The gate actually applied, in the selected rule's own units. */
	private static double activeRfpGate(final ClassifyRequest request) {
		return "legacy".equals(request.rfpMethod)
			? request.rfpLegacyThreshold : request.rfpNeighbourThreshold;
	}

	/**
	 * One block per image: counts, where its results went, whether they were
	 * computed or re-loaded, the three reprogramming ratios, and any non-fatal
	 * problem.
	 *
	 * A folder run is hours and mostly unattended, so the Log has to be the
	 * record of what happened to every image -- including the ones that failed,
	 * which a batch deliberately does not abort on.
	 */
	private static void logBatch(final Task task, final List<Map<String, Object>> images) {
		if (images == null) return;
		for (final Map<String, Object> image : images) {
			final String name = String.valueOf(image.get("image_name"));
			final String status = String.valueOf(image.get("status"));
			if ("failed".equals(status)) {
				IJ.log("Neural Image Classifier: " + name + " FAILED -- " + image.get("detail"));
				continue;
			}
			IJ.log("Neural Image Classifier: " + name
				+ " -- " + image.get("n_cells") + " cells"
				+ ", OPC+ " + image.get("n_opc_pos")
				+ ", B3Tub+ " + image.get("n_b3tub_pos")
				+ ", RFP+ " + image.get("n_rfp_pos")
				+ ("cached".equals(status) ? "  (loaded saved results)" : "")
				+ "  -> " + image.get("out_dir"));
			final Object detail = image.get("detail");
			if (detail != null && !String.valueOf(detail).isBlank()) {
				IJ.log("Neural Image Classifier:   " + detail);
			}
			@SuppressWarnings("unchecked")
			final List<String> quant = (List<String>) image.get("quantification");
			if (quant != null) {
				for (final String line : quant) IJ.log("Neural Image Classifier:   " + line);
			}
			@SuppressWarnings("unchecked")
			final List<String> warnings = (List<String>) image.get("warnings");
			if (warnings != null) {
				for (final String w : warnings) {
					IJ.log("Neural Image Classifier:   WARNING " + w);
				}
			}
		}
		@SuppressWarnings("unchecked")
		final List<String> summary = (List<String>) task.outputs.get("batch_summary");
		if (summary != null && !summary.isEmpty()) {
			IJ.log("Neural Image Classifier: " + summary.get(summary.size() - 1));
		}
	}

	/**
	 * Opens the figures of a single-image run; for a folder run, logs where they
	 * are instead.
	 *
	 * Twenty images would otherwise open sixty windows on top of the user's
	 * desktop at the end of an unattended overnight run. The files are written
	 * either way, one folder per image.
	 */
	private static void showImageFigures(
		final List<Map<String, Object>> images, final ClassifyRequest request
	) {
		if (images == null) return;
		final boolean open = images.size() == 1;
		for (final Map<String, Object> image : images) {
			final Object upset = image.get("upset_plot_path");
			if (upset != null && !String.valueOf(upset).isBlank()) {
				if (open) openFigure(String.valueOf(upset));
				else IJ.log("Neural Image Classifier: UpSet plot -> " + upset);
			}
			if (!request.showCaseMontage) continue;
			@SuppressWarnings("unchecked")
			final List<String> montages = (List<String>) image.get("case_montage_paths");
			if (montages == null) continue;
			for (final String path : montages) {
				if (open) openFigure(path);
				else IJ.log("Neural Image Classifier: case montage -> " + path);
			}
		}
	}

	/** Opens a saved figure. A figure that will not open must never read as a
	 *  failed run -- the classification results are already on screen. */
	private static void openFigure(final String path) {
		final ImagePlus imp = IJ.openImage(path);
		if (imp != null) imp.show();
		IJ.log("Neural Image Classifier: saved figure -> " + path);
	}


	/** Build (first call) or reuse (later calls) the Appose Python environment + worker. */
	private synchronized Service getPythonService() throws IOException, BuildException {
		if (environment == null) {
			final String envSpec = readResource("/scripts/environment.yml");
			environment = Appose.mamba()
				.content(envSpec)
				.subscribeProgress(this::showProgress)
				.subscribeOutput(IJ::log)
				.subscribeError(IJ::log)
				.build();
			hideProgress();
		}
		if (pythonService == null) {
			pythonService = environment.python();
		}
		return pythonService;
	}

	/**
	 * Renders one run's cells: always the Results table + CSV, and the ROI
	 * Manager only when asked for.
	 *
	 * The ROIs are bounding boxes in *patch* coordinates and the plugin never
	 * opens the CZI in Fiji, so there is nothing on screen for them to overlay;
	 * forcing the ROI Manager window open at the end of every run was noise for
	 * anyone not using them. Ticking the box still gets them, which is what makes
	 * "Show Cell..." able to pre-fill a cell id from the current selection.
	 */
	private void renderResults(
		final List<Map<String, Object>> cells, final ClassifyRequest request
	) {
		if (cells == null || cells.isEmpty()) {
			IJ.log("Neural Image Classifier: no cells to show.");
			return;
		}

		RoiManager rm = null;
		if (request.addRois) {
			rm = RoiManager.getInstance();
			if (rm == null) rm = new RoiManager();
			rm.reset();
		}

		final ResultsTable rt = new ResultsTable();

		for (final Map<String, Object> c : cells) {
			final String imageName = String.valueOf(c.get("image_name"));
			final int y0 = ((Number) c.get("bb_y0")).intValue();
			final int y1 = ((Number) c.get("bb_y1")).intValue();
			final int x0 = ((Number) c.get("bb_x0")).intValue();
			final int x1 = ((Number) c.get("bb_x1")).intValue();
			final boolean opcPos = (Boolean) c.get("opc_pos");
			final boolean b3tubPos = (Boolean) c.get("b3tub_pos");
			final boolean rfpPos = (Boolean) c.get("rfp_pos");

			// Bounding-box ROI for a first working version; a follow-up can
			// return each cell's native_mask from Python (as an Appose
			// NDArray) and build a pixel-precise polygon/mask ROI instead.
			if (rm != null) {
				final Roi roi = new Roi(x0, y0, x1 - x0, y1 - y0);
				roi.setStrokeColor(roiColor(opcPos, b3tubPos));
				roi.setName(roiName(imageName, ((Number) c.get("cell_id")).intValue()));
				rm.addRoi(roi);
			}

			rt.incrementCounter();
			// The image name leads every row: cell_id restarts at 0 for each
			// image, so it is only unique together with this column -- in the
			// table, in the ROI names, and in "Show Cell...".
			rt.addValue("image", imageName);
			rt.addValue("cell_id", ((Number) c.get("cell_id")).doubleValue());
			rt.addValue("patch_idx", ((Number) c.get("patch_idx")).doubleValue());
			rt.addValue("opc_prob", ((Number) c.get("opc_prob")).doubleValue());
			rt.addValue("opc_pos", opcPos ? 1 : 0);
			rt.addValue("b3tub_prob", ((Number) c.get("b3tub_prob")).doubleValue());
			rt.addValue("b3tub_pos", b3tubPos ? 1 : 0);
			rt.addValue("rfp_pos", rfpPos ? 1 : 0);
			// Both RFP rules travel with every cell, whichever one made the
			// call, so the table can be re-read under the other one without
			// re-running the image. They are NOT on the same scale -- different
			// background estimators -- so a single gate cannot be compared
			// across the two columns.
			rt.addValue("rfp_method", String.valueOf(c.get("rfp_method")));
			rt.addValue("rfp_neighbour_score",
				((Number) c.get("rfp_neighbour_score")).doubleValue());
			rt.addValue("rfp_neighbour_pos", (Boolean) c.get("rfp_neighbour_pos") ? 1 : 0);
			rt.addValue("rfp_legacy_pos", (Boolean) c.get("rfp_legacy_pos") ? 1 : 0);
			rt.addValue("rfp_score", ((Number) c.get("rfp_score")).doubleValue());
			rt.addValue("opc_channel_score", ((Number) c.get("opc_channel_score")).doubleValue());
			rt.addValue("b3tub_channel_score",
				((Number) c.get("b3tub_channel_score")).doubleValue());
			rt.addValue("n_cells_in_region",
				((Number) c.get("n_cells_in_region")).doubleValue());
		}

		rt.show("Neural Classifier Results");

		// Python already wrote each image's own CSV into that image's output
		// subfolder -- it has every column and survives a batch nobody watches.
		// This extra file is the combined view, matching what is on screen.
		final Path csvPath = Paths.get(request.outDir, "all_images_classifications.csv");
		try {
			rt.saveAs(csvPath.toString());
			IJ.log("Neural Image Classifier: saved combined results -> " + csvPath);
		}
		catch (final IOException e) {
			IJ.handleException(e);
		}
	}

	/** ROI name carrying both halves of a cell's identity. */
	private static String roiName(final String imageName, final int cellId) {
		return stripExtension(imageName) + "__cell_" + cellId;
	}

	private static String stripExtension(final String fileName) {
		return fileName.replaceFirst("\\.[^.]+$", "");
	}

	/** OPC+/B3Tub+ (magenta) is the biologically ambiguous double-positive case
	 *  that the notebook flags for manual review -- keep it visually distinct. */
	private static Color roiColor(final boolean opcPos, final boolean b3tubPos) {
		if (opcPos && b3tubPos) return Color.MAGENTA;
		if (opcPos) return Color.CYAN;
		if (b3tubPos) return Color.YELLOW;
		return Color.GRAY;
	}

	// -- "Show Cell (all channels)..." : separate, on-demand menu action --

	/** {@code <image stem>__cell_<id>} -- see {@link #roiName}. */
	private static final Pattern CELL_ROI_NAME = Pattern.compile("(.+)__cell_(\\d+)");

	/**
	 * Prompts for (or infers from the current ROI Manager selection) a cell from
	 * the last run, fetches its raw 5-channel crop from the Python worker, and
	 * shows it as a composite hyperstack.
	 *
	 * A cell is identified by image AND id: a batch numbers each image's cells
	 * from 0, so an id alone is ambiguous as soon as more than one image was
	 * processed. The image chooser only appears when there is a choice to make.
	 */
	private void showCellDialog() throws IOException, BuildException, InterruptedException, TaskException {
		if (lastCells == null || lastCells.isEmpty()) {
			IJ.error(
				"Neural Image Classifier",
				"No classification results yet.\nRun 'Classify Cells (OPC/B3Tub/RFP)...' first."
			);
			return;
		}

		final List<String> imageNames = classifiedImageNames();
		final String[] selection = inferSelectedCell();
		String imageName = selection != null ? selection[0] : imageNames.get(0);
		if (!imageNames.contains(imageName)) imageName = imageNames.get(0);
		final int defaultId = selection != null
			? Integer.parseInt(selection[1])
			: ((Number) lastCells.get(0).get("cell_id")).intValue();

		final GenericDialog gd = new GenericDialog("Show Cell");
		gd.addMessage("Tip: select a single ROI in the ROI Manager first to pre-fill it.");
		if (imageNames.size() > 1) {
			gd.addChoice("Image:", imageNames.toArray(new String[0]), imageName);
		}
		gd.addNumericField("Cell id:", defaultId, 0);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		if (imageNames.size() > 1) imageName = gd.getNextChoice();
		final int cellId = (int) gd.getNextNumber();

		final Map<String, Object> cell = findCell(imageName, cellId);
		if (cell == null) {
			IJ.error("Neural Image Classifier",
				"No cell with id " + cellId + " in " + imageName + ".");
			return;
		}

		showCell(cell);
	}

	/** Image names present in the last run, in first-seen (i.e. input) order. */
	private static List<String> classifiedImageNames() {
		final List<String> names = new ArrayList<>();
		for (final Map<String, Object> c : lastCells) {
			final String name = String.valueOf(c.get("image_name"));
			if (!names.contains(name)) names.add(name);
		}
		return names;
	}

	/** {@code {imageName, cellId}} of the single selected ROI, or null.
	 *
	 *  The ROI carries the image *stem*, so it is matched back against the full
	 *  names by prefix rather than assuming an extension.
	 */
	private static String[] inferSelectedCell() {
		final RoiManager rm = RoiManager.getInstance();
		if (rm == null) return null;
		final int[] selected = rm.getSelectedIndexes();
		if (selected.length != 1) return null;
		final Matcher m = CELL_ROI_NAME.matcher(rm.getName(selected[0]));
		if (!m.matches()) return null;
		final String stem = m.group(1);
		for (final String name : classifiedImageNames()) {
			if (stripExtension(name).equals(stem)) return new String[] { name, m.group(2) };
		}
		return null;
	}

	private static Map<String, Object> findCell(final String imageName, final int cellId) {
		for (final Map<String, Object> c : lastCells) {
			if (((Number) c.get("cell_id")).intValue() == cellId
				&& imageName.equals(String.valueOf(c.get("image_name"))))
			{
				return c;
			}
		}
		return null;
	}

	private void showCell(final Map<String, Object> cell)
		throws IOException, BuildException, InterruptedException, TaskException
	{
		final Service python = getPythonService();

		final Map<String, Object> inputs = new HashMap<>();
		// Which image, not just which cell: a batch leaves cells from several
		// CZIs in the table, and the worker keeps only one decoded image.
		inputs.put("image_path", String.valueOf(cell.get("image_path")));
		inputs.put("patch_idx", ((Number) cell.get("patch_idx")).intValue());
		inputs.put("bb_y0", ((Number) cell.get("bb_y0")).intValue());
		inputs.put("bb_y1", ((Number) cell.get("bb_y1")).intValue());
		inputs.put("bb_x0", ((Number) cell.get("bb_x0")).intValue());
		inputs.put("bb_x1", ((Number) cell.get("bb_x1")).intValue());

		final String script = readResource("/scripts/get_cell_crop.py");
		final Task task = python.task(script, inputs);
		// On a cached run the CZI has not been read yet; this task does it, once.
		task.listen(event -> {
			if (event.responseType == ResponseType.UPDATE) {
				IJ.log("Neural Image Classifier: " + event.message);
			}
		});
		task.start();
		task.waitFor();
		if (task.status != TaskStatus.COMPLETE) {
			throw new RuntimeException("Python task failed: " + task.error);
		}

		final NDArray crop = (NDArray) task.outputs.get("cell_crop");
		@SuppressWarnings("unchecked")
		final List<String> channelNames = (List<String>) task.outputs.get("channel_names");

		final int[] shape = crop.shape().toIntArray(); // [C, H, W], C-order
		final int nChannels = shape[0], h = shape[1], w = shape[2];
		final ShortBuffer pixels = crop.buffer().order(ByteOrder.nativeOrder()).asShortBuffer();

		final ImageStack stack = new ImageStack(w, h);
		for (int c = 0; c < nChannels; c++) {
			final short[] plane = new short[w * h];
			pixels.position(c * w * h);
			pixels.get(plane);
			final String label = c < channelNames.size() ? channelNames.get(c) : ("ch" + c);
			stack.addSlice(label, new ShortProcessor(w, h, plane, null));
		}

		final int cellId = ((Number) cell.get("cell_id")).intValue();
		final ImagePlus imp = new ImagePlus(
			roiName(String.valueOf(cell.get("image_name")), cellId), stack);
		imp.setDimensions(nChannels, 1, 1);
		final CompositeImage comp = new CompositeImage(imp, CompositeImage.COMPOSITE);
		for (int c = 0; c < nChannels; c++) {
			comp.setChannelLut(LUT.createLutFromColor(channelColor(channelNames.get(c))), c + 1);
		}
		// resetDisplayRange() only rescales the CURRENT channel, so it has to be
		// driven once per channel: a channel left at its initial 0..0 range renders
		// every pixel above its max, i.e. a solid saturated block. These are raw
		// uint16 CZI values occupying a small part of the 16-bit range, and each
		// channel's brightness differs, so per-channel autoscaling is also what
		// makes them individually readable.
		for (int c = 1; c <= nChannels; c++) {
			comp.setC(c);
			comp.resetDisplayRange();
		}
		comp.setC(1);
		comp.show();
	}

	/** Cosmetic per-channel display color; DAPI/BF/OPC/RFP/B3Tub is the fixed
	 *  channel order the pipeline reads CZIs in (see CHANNEL_NAMES in Python). */
	private static Color channelColor(final String channelName) {
		switch (channelName) {
			case "DAPI": return Color.BLUE;
			case "OPC": return Color.GREEN;
			case "RFP": return Color.RED;
			case "B3Tub": return Color.MAGENTA;
			default: return Color.WHITE; // BF and anything unrecognized: grayscale
		}
	}

	// -- models: fetched once, then found on every later run -----------------

	/**
	 * Makes sure {@code modelDir} holds the checkpoints, downloading the
	 * published bundle if it does not.
	 *
	 * A no-op after the first run, which is why this sits on the normal run path
	 * rather than behind a setup step the user has to remember: the only way to
	 * be wrong about whether the models are installed is to look.
	 */
	private void installModels(final Path modelDir) throws IOException {
		if (ModelBootstrap.isInstalled(modelDir)) return;
		progressPrefix = "Downloading models";
		try {
			final boolean downloaded = ModelBootstrap.ensureInstalled(
				modelDir, this::showProgress);
			if (downloaded) {
				IJ.log("Neural Image Classifier: installed model bundle "
					+ ModelBootstrap.BUNDLE_VERSION + " -> " + modelDir);
			}
		}
		finally {
			hideProgress();
			progressPrefix = "Building Python environment";
		}
	}

	/**
	 * {@code Install / Verify Models...}: the same install the first run does,
	 * on demand.
	 *
	 * Worth its own menu entry because the download is ~90 MB and a classify run
	 * is hours: someone setting a laptop up before a session, or on a connection
	 * they do not want to depend on later, should be able to get the models in
	 * place and confirm it without starting a run. It also answers "are the
	 * models actually there?", which otherwise has no answer short of trying.
	 */
	private void installModelsDialog() throws IOException {
		final GenericDialog gd = new GenericDialog("Install / Verify Models");
		gd.addMessage("Neural Image Classifier " + BUILD_ID + " -- model bundle "
			+ ModelBootstrap.BUNDLE_VERSION, SECTION_FONT);
		gd.addDirectoryField("Model directory:", DEFAULT_MODEL_DIR);
		addHint(gd, "Status is checked when you press OK. An existing, complete "
			+ "directory is left untouched.");
		gd.addMessage("OK downloads the OPC + B3-Tub checkpoints and thresholds.json "
			+ "(about 90 MB) into that directory,\nunless they are already there. "
			+ "Needs internet the first time only.", HINT_FONT, HINT_COLOR);
		gd.showDialog();
		if (gd.wasCanceled()) return;

		final Path modelDir = Paths.get(gd.getNextString().trim());
		IJ.log("Neural Image Classifier: " + ModelBootstrap.describe(modelDir));
		if (ModelBootstrap.isInstalled(modelDir)) {
			IJ.showMessage("Install / Verify Models",
				"Models are already installed in:\n" + modelDir
				+ "\n\nNothing to download.");
			return;
		}
		try {
			installModels(modelDir);
		}
		catch (final IOException e) {
			// Shown, not just thrown: this command exists so a user can find out
			// whether the models are in place, and an exception trace in the Log
			// does not answer that question for the person who ran it.
			IJ.error("Install / Verify Models", "Model download failed.\n\n"
				+ e.getMessage() + "\n\nBundle URL:\n" + ModelBootstrap.bundleUrl());
			return;
		}
		IJ.showMessage("Install / Verify Models",
			"Models installed in:\n" + modelDir + "\n\nYou can now run "
			+ "Classify Cells.");
	}

	/**
	 * The closest ancestor of {@code path} that exists -- what a writability
	 * check has to ask about, since {@code Files.isWritable} on a path that does
	 * not exist yet is always false and would reject every fresh install.
	 */
	private static Path nearestExistingAncestor(final Path path) {
		for (Path p = path.toAbsolutePath(); p != null; p = p.getParent()) {
			if (Files.exists(p)) return p;
		}
		return null;
	}

	// -- progress dialog: building the conda env on first run takes minutes --

	private JDialog progressDialog;
	private JProgressBar progressBar;

	/**
	 * What the progress bar is currently reporting on.
	 *
	 * The same window serves two different multi-minute first-run steps -- the
	 * model download and the conda environment build. Without this the bar would
	 * label a 90 MB model download "Building Python environment", which is the
	 * one moment a user is most likely to be deciding whether it has hung.
	 */
	private String progressPrefix = "Building Python environment";

	private void showProgress(final String title, final long current, final long maximum) {
		EventQueue.invokeLater(() -> {
			if (progressDialog == null) {
				final Window owner = IJ.getInstance();
				progressDialog = new JDialog(owner, "Neural Image Classifier");
				progressDialog.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
				progressBar = new JProgressBar();
				progressBar.setFont(new Font("Courier", Font.PLAIN, 14));
				progressBar.setStringPainted(true);
				progressBar.setIndeterminate(true);
				progressDialog.getContentPane().add(progressBar);
				progressDialog.pack();
				progressDialog.setLocationRelativeTo(owner);
				progressDialog.setVisible(true);
			}
			if (maximum > 0) {
				progressBar.setIndeterminate(false);
				progressBar.setMinimum(0);
				progressBar.setMaximum((int) maximum);
				progressBar.setValue((int) current);
			}
			final String label = (title == null || title.trim().isEmpty()) ? "" : title.trim();
			progressBar.setString(label.isEmpty() ? progressPrefix
				: progressPrefix + ": " + label);
		});
	}

	private void hideProgress() {
		EventQueue.invokeLater(() -> {
			if (progressDialog != null) {
				progressDialog.dispose();
				progressDialog = null;
				progressBar = null;
			}
		});
	}

	private static String readResource(final String path) throws IOException {
		try (InputStream is = NeuralClassifyPlugin.class.getResourceAsStream(path)) {
			if (is == null) throw new IOException("Resource not found: " + path);
			return new String(is.readAllBytes(), StandardCharsets.UTF_8);
		}
	}

	/** Lets the plugin be launched directly from an IDE for manual testing. */
	public static void main(final String[] args) {
		ij.ImageJ.main(args);
		new NeuralClassifyPlugin().run("");
	}
}
