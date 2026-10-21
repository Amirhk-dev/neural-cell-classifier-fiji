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
	static final String BUILD_ID = "1.2.0";

	private static final int DEFAULT_PATCH_GRID = 4;

	/** Channel roles, in dialog order. Must match the role names
	 *  {@code neural_imgs.channels} accepts: this side only ever sends the
	 *  canonical spellings, which is why the alias table that module keeps for
	 *  hand-written layouts has no second copy here to drift from. */
	private static final String ROLE_DAPI = "DAPI", ROLE_OPC = "OPC",
		ROLE_RFP = "RFP", ROLE_B3TUB = "B3Tub", ROLE_BF = "BF";
	private static final String[] CHANNEL_ROLES =
		{ ROLE_DAPI, ROLE_OPC, ROLE_RFP, ROLE_B3TUB, ROLE_BF };
	/** Dialog captions, one per role. Short because the five channel fields
	 *  share one row -- the "Channels" heading already says these are channel
	 *  numbers, so repeating "channel" five times only widened the dialog. */
	private static final String[] CHANNEL_CAPTIONS =
		{ "DAPI", "OPC", "RFP", "B3-Tub", "BF" };

	/** Marker roles a cell is called positive in, i.e. the ones whose absence
	 *  removes columns from the results table. A subset of CHANNEL_ROLES. */
	private static final String[] MARKER_ROLES = { ROLE_OPC, ROLE_B3TUB, ROLE_RFP };

	/** Default channel numbers, 1-based, in CHANNEL_ROLES order -- the fixed
	 *  acquisition every previous build assumed. */
	private static final int[] DEFAULT_CHANNELS = { 1, 2, 3, 4, 5 };

	/** What a channel field is set to when the image does not have that channel.
	 *  Only the markers may be absent; DAPI cannot (see validate). */
	private static final int CHANNEL_ABSENT = 0;

	/** Upper bound on a channel number, so a typo cannot ask for channel 900 and
	 *  be reported as "the image has only 5" minutes later. Well above any
	 *  plausible widefield acquisition. */
	private static final int MAX_CHANNEL = 32;

	/** Width of a channel field, in characters. Two digits is every value
	 *  {@link #MAX_CHANNEL} allows, and narrow fields are what let all five
	 *  share one row. */
	private static final int CHANNEL_FIELD_COLUMNS = 2;

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

	/** Field hints: one short grey italic line under the field they describe.
	 *  Deliberately one line each -- anything that needs a paragraph belongs in
	 *  Help, where it can be read without making every run's dialog taller.
	 *  The single-image and folder hints are greyed out together with their
	 *  field when the other input mode is selected (see InputRow.setEnabled). */
	private static final String HINT_IMAGE = "One Zeiss .czi file.";
	private static final String HINT_FOLDER =
		"Every .czi directly inside it; not recursive.";
	/** One line for both directory rows: neither is ever greyed out, so they do
	 *  not need separate handles, and two lines of grey italics to say two short
	 *  things made the dialog taller than the things were worth. */
	private static final String HINT_DIRS =
		"Models download on first run; the output root gets one subfolder per image.";
	/** Sits under the "Channels" heading rather than under a field, because it
	 *  describes all five at once. */
	private static final String HINT_CHANNELS =
		"Channel number of each marker, counting from 1. "
		+ "0 = not in my images; DAPI is required.";

	/** Hint styling. The disabled colour is what marks a hint as "not this mode":
	 *  MultiLineLabel paints its own text and ignores setEnabled, so the greying
	 *  has to be done through the foreground colour. */
	private static final Font HINT_FONT = new Font("SansSerif", Font.ITALIC, 11);
	/** Group headings. Bold, so the required block at the top of the dialog is
	 *  visibly one group and not just the first few rows. */
	private static final Font SECTION_FONT = new Font("SansSerif", Font.BOLD, 12);
	private static final Color HINT_COLOR = new Color(0x55, 0x55, 0x55);
	private static final Color HINT_DISABLED_COLOR = new Color(0xAA, 0xAA, 0xAA);

	/** Where Help sends anyone who wants more than the dialog explains. The
	 *  repository front page, not a Pages site, so the link cannot rot ahead of
	 *  the docs being published; point it at the Pages URL once there is one. */
	private static final String DOCS_URL =
		"https://github.com/Amirhk-dev/neural-cell-classifier-fiji";
	private static final String ISSUES_URL = DOCS_URL + "/issues";

	/** Shown by the dialog's Help button. ImageJ renders a help string that
	 *  starts with {@code <html>} in its own window instead of opening a
	 *  browser, so the plugin stays self-contained offline.
	 *
	 *  Kept to what the dialog itself cannot say: the one-time downloads, what
	 *  0 means in Channels, and why there are two RFP gates. Anything longer --
	 *  the saved-results key, the RFP evidence, the per-marker output rules --
	 *  lives in the manual linked at the bottom, rather than in a window a user
	 *  has to scroll past to reach the sentence they came for. */
	private static final String DIALOG_HELP =
		"<html><body style='font-family:sans-serif;width:460px'>"
		+ "<h2 style='font-size:14pt;margin:0'>Neural Image Classifier</h2>"
		+ "<p style='color:#555;margin:2px 0 12px 0;font-size:10pt'>"
		+ "Single-cell OPC / Neuron (B3-Tub) / Transduced (RFP) calls on Zeiss "
		+ "CZI images &nbsp;&middot;&nbsp; build " + BUILD_ID + "</p>"

		+ "<p><b>One run:</b> the CZI is tiled <i>Patch grid</i> &times; <i>Patch "
		+ "grid</i>, illumination-corrected (BaSiC), nuclei are detected on DAPI "
		+ "(CellPose), and every cell is scored by the two ResNet-18 classifiers "
		+ "and the RFP rule &mdash; for whichever markers your image declares a "
		+ "channel. You get the <i>Neural Classifier Results</i> table, a CSV, an "
		+ "UpSet plot, and optionally ROIs and example cells. The dialog reopens "
		+ "pre-filled after each run; <b>Cancel</b> finishes.</p>"

		+ "<p>Only the first group is required. Everything below it is tuning, "
		+ "already set to the published defaults &mdash; except <b>Channels</b>, "
		+ "which describes <i>your</i> images.</p>"

		+ "<h3 style='margin-bottom:2px'>First run downloads ~1&ndash;2 GB, once</h3>"
		+ "<p style='margin-top:0'>The Python environment (PyTorch, CellPose, "
		+ "BaSiCPy) and the trained models (~90 MB) are fetched and cached; expect "
		+ "several minutes and leave the progress window alone. Later runs start "
		+ "in seconds. <i>Plugins &gt; Neural Image Processor &gt; Install / "
		+ "Verify Models...</i> does the model half ahead of time, or just reports "
		+ "what is already installed.</p>"

		+ "<h3 style='margin-bottom:2px'>Channels: 0 means \"not in my images\"</h3>"
		+ "<p style='margin-top:0'>One field per marker, holding its channel "
		+ "number in your image counting from 1. Set a marker to <b>0</b> and the "
		+ "run still goes ahead: that marker is reported as <b>not measured</b> "
		+ "&mdash; no column, no UpSet set, no ratio &mdash; which is deliberately "
		+ "different from <i>measured and negative</i>. So DAPI + RFP + B3-Tub and "
		+ "no OPC is <code>1, 0, 2, 3, 0</code>, and it runs.</p>"
		+ "<p><b>DAPI cannot be 0</b> &mdash; every cell in the run comes from the "
		+ "nucleus detection on it. Any order is fine, gaps are fine, and channels "
		+ "above the highest number you use need not be mentioned at all.</p>"

		+ "<h3 style='margin-bottom:2px'>Defaults</h3>"
		+ "<table cellpadding='3' cellspacing='0' style='font-size:10pt'>"
		+ "<tr><td><b>Patch grid</b></td><td><code>4</code></td>"
		+ "<td>tiles per side, so 16 tiles. A count, not a pixel size.</td></tr>"
		+ "<tr><td><b>Cell probability</b></td><td><code>0.1</code></td>"
		+ "<td>&minus;3.0&hellip;2.0. Lower &rarr; more nuclei detected.</td></tr>"
		+ "<tr><td><b>Flow</b></td><td><code>0.2</code></td>"
		+ "<td>0.0&hellip;1.0. Lower &rarr; fewer masks accepted.</td></tr>"
		+ "<tr><td><b>RFP rule</b></td><td><code>neighbour</code></td>"
		+ "<td>gate <code>16.5</code>, exclusion radius <code>60</code> px.</td></tr>"
		+ "<tr><td><b>Images ahead</b></td><td><code>1</code></td>"
		+ "<td>the memory knob &mdash; each waiting image holds its whole tile "
		+ "array.</td></tr>"
		+ "<tr><td><b>Threads</b></td><td><code>0</code></td>"
		+ "<td>cell-extraction pool; 0 sizes it from this process's cores.</td></tr>"
		+ "</table>"

		+ "<h3 style='margin-bottom:2px'>RFP has no classifier</h3>"
		+ "<p style='margin-top:0'>It is called from pixel intensity. "
		+ "<b>neighbour</b> (production) compares the cell against its crop's "
		+ "background with pixels near other nuclei dropped; <b>legacy</b> uses "
		+ "the crop's 95th percentile. Both are scored for every cell and both "
		+ "travel in the results table, so a saved table can be re-read under the "
		+ "other rule without re-classifying.</p>"
		+ "<p style='color:#a00'><b>The two gates are not interchangeable</b> "
		+ "&mdash; different background estimators put the scores on different "
		+ "scales, which is why each rule has its own field.</p>"

		+ "<h3 style='margin-bottom:2px'>Saved results</h3>"
		+ "<p style='margin-top:0'>With <i>Load saved results</i> ticked, an image "
		+ "whose previous run matches on everything that could change the numbers "
		+ "&mdash; the image, patch grid, channels, both thresholds, the RFP "
		+ "settings and the model bundle &mdash; is re-read in seconds instead of "
		+ "re-classified. Any mismatch re-runs it, and the Log names what "
		+ "changed.</p>"

		+ "<hr>"
		+ "<p style='font-size:10pt'><b>Manual:</b> "
		+ "<a href='" + DOCS_URL + "'>" + DOCS_URL + "</a><br>"
		+ "<b>Report a problem:</b> <a href='" + ISSUES_URL + "'>" + ISSUES_URL
		+ "</a> &mdash; please quote build " + BUILD_ID + " and the Fiji Log of "
		+ "the failing run.<br>"
		+ "<b>Licence:</b> MIT &copy; 2025 Amirhossein Kardoost. CellPose, BaSiC, "
		+ "PyTorch and Fiji/ImageJ are used under their own licences.</p>"
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
		/** 1-based channel number per {@link #CHANNEL_ROLES} entry, or
		 *  {@link #CHANNEL_ABSENT} for a channel this image does not have.
		 *  Turned into the {@code channel_order} string Python parses by
		 *  {@link #channelOrder}; kept as numbers here because that is what the
		 *  dialog collects and what can be validated without a parser. */
		final int[] channels;
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
			final int[] channels,
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
			this.channels = channels.clone();
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

		/** The channel number given for {@code role}, or {@link #CHANNEL_ABSENT}. */
		int channelOf(final String role) {
			for (int i = 0; i < CHANNEL_ROLES.length; i++) {
				if (CHANNEL_ROLES[i].equals(role)) return channels[i];
			}
			return CHANNEL_ABSENT;
		}

		boolean measures(final String role) {
			return channelOf(role) != CHANNEL_ABSENT;
		}

		/**
		 * The channel assignment as the {@code channel_order} string Python parses:
		 * one role per image channel, in file order, {@code "-"} for a channel this
		 * run does not use -- e.g. {@code "DAPI,-,RFP,B3Tub"}.
		 *
		 * Only as long as the highest channel actually assigned. Channels past
		 * that are not listed at all, which Python reads as "present in the file
		 * and ignored" -- so a 5-channel image whose last two channels are unused
		 * needs no declaration about them, and the same assignment also fits a
		 * 3-channel one.
		 */
		String channelOrder() {
			int highest = 0;
			for (final int channel : channels) highest = Math.max(highest, channel);
			final String[] roles = new String[highest];
			for (int i = 0; i < highest; i++) roles[i] = "-";
			for (int i = 0; i < CHANNEL_ROLES.length; i++) {
				if (channels[i] != CHANNEL_ABSENT) roles[channels[i] - 1] = CHANNEL_ROLES[i];
			}
			return String.join(",", roles);
		}

		/** Marker roles this run measures, for the results table and the Log. */
		List<String> markers() {
			final List<String> out = new ArrayList<>();
			for (final String role : MARKER_ROLES) {
				if (measures(role)) out.add(role);
			}
			return out;
		}

		/** One line for the Log: the assignment, and what it leaves unmeasured. */
		String describeChannels() {
			final StringBuilder sb = new StringBuilder("channels " + channelOrder());
			final List<String> missing = new ArrayList<>();
			for (final String role : MARKER_ROLES) {
				if (!measures(role)) missing.add(role);
			}
			if (!missing.isEmpty()) {
				sb.append(", not measured: ").append(String.join(", ", missing));
			}
			return sb.toString();
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
		addSection(gd, "Input and output");
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
		gd.addDirectoryField("Output directory:",
			previous == null ? System.getProperty("user.home") : previous.outDir);
		addHint(gd, HINT_DIRS);

		// Immediately after the required block, because it is the one thing a
		// new dataset is most likely to need changed -- and getting it wrong
		// means scoring the wrong pixels rather than failing, so it should not
		// be buried among the tuning sections.
		addSection(gd, "Channels");
		// Above the fields, not below: it explains all five at once, and a
		// five-field row reads as a single question with one caption over it.
		addHint(gd, HINT_CHANNELS);
		for (int i = 0; i < CHANNEL_ROLES.length; i++) {
			// All five on one row. Each is a one- or two-digit channel number,
			// so a row apiece spent five rows of dialog height on ten
			// characters of input -- and side by side they also read as the one
			// thing they are: a layout.
			if (i > 0) gd.addToSameRow();
			gd.addNumericField(CHANNEL_CAPTIONS[i],
				previous == null ? DEFAULT_CHANNELS[i] : previous.channels[i],
				0, CHANNEL_FIELD_COLUMNS, "");
		}

		addSection(gd, "Detection");
		// Patch grid is a tile count -> 0 decimals; the two thresholds are
		// genuinely continuous, so they get decimal places. The digit count only
		// formats the default shown -- ImageJ parses whatever is typed, at full
		// precision, either way, so validate() is what actually enforces
		// "whole number" and the allowed ranges.
		gd.addNumericField("Patch grid (NxN):",
			previous == null ? DEFAULT_PATCH_GRID : previous.patchGrid, 0, 3, "");
		gd.addToSameRow();
		gd.addNumericField("Cell probability:",
			previous == null ? DEFAULT_CELLPROB_THRESHOLD : previous.cellprobThreshold,
			2, 5, "");
		gd.addToSameRow();
		gd.addNumericField("Flow:",
			previous == null ? DEFAULT_FLOW_THRESHOLD : previous.flowThreshold, 2, 5, "");

		addSection(gd, "RFP positivity (an intensity rule, not a classifier)");
		gd.addChoice("Rule:", RFP_METHODS,
			previous == null ? RFP_METHODS[0] : previous.rfpMethod);
		gd.addToSameRow();
		gd.addNumericField("Exclusion radius:",
			previous == null ? DEFAULT_RFP_EXCLUDE_RADIUS : previous.rfpExcludeRadius,
			0, 4, "px");
		// One field per rule, on one row: the two gates are NOT on the same
		// scale, so seeing them together is the point -- but they are still two
		// numbers and do not need two rows.
		gd.addNumericField("Gate, neighbour:",
			previous == null ? DEFAULT_RFP_NEIGHBOUR_THRESHOLD : previous.rfpNeighbourThreshold,
			2, 5, "");
		gd.addToSameRow();
		gd.addNumericField("Gate, legacy:",
			previous == null ? DEFAULT_RFP_LEGACY_THRESHOLD : previous.rfpLegacyThreshold,
			2, 5, "");

		addSection(gd, "Performance");
		gd.addNumericField("Images prepared ahead:",
			previous == null ? DEFAULT_PREFETCH : previous.prefetch, 0, 3, "");
		gd.addToSameRow();
		gd.addNumericField("Extraction threads:",
			previous == null ? DEFAULT_WORKERS : previous.workers, 0, 3, "0 = auto");

		addSection(gd, "Outputs");
		// Two by two. Four full-width checkboxes carrying a parenthesised
		// explanation each were the tallest and widest block in the dialog, for
		// four on/off switches; what the parentheses said is in Help.
		gd.addCheckbox("UpSet plot", previous == null || previous.generateUpset);
		gd.addToSameRow();
		gd.addCheckbox("Example cell per UpSet case",
			previous != null && previous.showCaseMontage);
		gd.addCheckbox("Cell ROIs in the ROI Manager",
			previous != null && previous.addRois);
		gd.addToSameRow();
		gd.addCheckbox("Load saved results if available",
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
		// of that type silently shifts by one. addToSameRow() only changes where
		// a field is painted, never that order, so the fields sharing a row are
		// still read one at a time, left to right.
		final boolean folderMode = gd.getNextChoiceIndex() == MODE_FOLDER;
		final String imagePath = gd.getNextString();
		final String inputDir = gd.getNextString();
		final String modelDir = gd.getNextString();
		final String outDir = gd.getNextString();
		// The channel fields are read here because they were *added* here --
		// getNext* is positional per field type, so this block must stay in step
		// with the dialog above it.
		final double[] channelEntries = new double[CHANNEL_ROLES.length];
		for (int i = 0; i < channelEntries.length; i++) {
			channelEntries[i] = gd.getNextNumber();
		}
		// Read as a double, not (int): the cast alone would silently turn a
		// typed "4.7" into 4 rather than telling the user it was ignored.
		final double patchGridEntry = gd.getNextNumber();
		final double cellprobThreshold = gd.getNextNumber();
		final double flowThreshold = gd.getNextNumber();
		final String rfpMethod = gd.getNextChoice();
		// Exclusion radius before the two gates, because that is the order the
		// RFP rows add them: the radius shares a row with the rule choice.
		final double rfpExcludeRadiusEntry = gd.getNextNumber();
		final double rfpNeighbourThreshold = gd.getNextNumber();
		final double rfpLegacyThreshold = gd.getNextNumber();
		final double prefetchEntry = gd.getNextNumber();
		final double workersEntry = gd.getNextNumber();
		final boolean generateUpset = gd.getNextBoolean();
		final boolean showCaseMontage = gd.getNextBoolean();
		final boolean addRois = gd.getNextBoolean();
		final boolean reuseCache = gd.getNextBoolean();

		// Out-of-range numbers are carried through to validate() rather than
		// rejected here, so a re-prompt can show them back with the complaint.
		// A fractional or NaN channel entry becomes an out-of-range int, which
		// validate() then names -- rather than being silently truncated to a
		// channel the user did not ask for.
		final int[] channels = new int[channelEntries.length];
		for (int i = 0; i < channels.length; i++) {
			channels[i] = Double.isNaN(channelEntries[i]) ? -1
				: (channelEntries[i] != Math.rint(channelEntries[i]) ? -1
					: (int) channelEntries[i]);
		}

		return new ClassifyRequest(
			folderMode, imagePath, inputDir, channels, (int) patchGridEntry,
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
	 *  group before.
	 *
	 *  The gap is what makes the groups legible now that most of them are one
	 *  or two rows -- without it a heading would sit as close to the group above
	 *  it as to its own. */
	private static void addSection(final GenericDialog gd, final String title) {
		gd.setInsets(10, 0, 2);   // applies to the next component only
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
		final String channelProblem = validateChannels(request);
		if (channelProblem != null) return channelProblem;
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
	 * First thing wrong with the channel assignment, phrased for the user, or null.
	 *
	 * Checked here rather than left to Python because it is the one input a user
	 * has to describe about their own data, and a mistake in it is worth a dialog
	 * now instead of a stack trace after the models load. What this cannot know
	 * is how many channels the image actually has -- that needs the file open, so
	 * {@code load_patches} checks it per image, before that image's BaSiC fit,
	 * and reports both counts.
	 */
	private static String validateChannels(final ClassifyRequest request) {
		for (int i = 0; i < CHANNEL_ROLES.length; i++) {
			final int channel = request.channels[i];
			if (channel < CHANNEL_ABSENT || channel > MAX_CHANNEL) {
				return "Channels: " + CHANNEL_CAPTIONS[i]
					+ "\n\nMust be a whole channel number between 1 and " + MAX_CHANNEL
					+ (ROLE_DAPI.equals(CHANNEL_ROLES[i]) ? "." : ", or 0 if the image "
						+ "has no " + CHANNEL_ROLES[i] + " channel.");
			}
		}
		if (request.channelOf(ROLE_DAPI) == CHANNEL_ABSENT) {
			return "The DAPI channel cannot be 0.\n\nNuclei are detected on DAPI, "
				+ "so without it there are no single cells to classify. Every other "
				+ "channel may be left out.";
		}
		// Two roles on one channel would mean scoring the same pixels twice under
		// different names, which is never what was meant.
		for (int i = 0; i < CHANNEL_ROLES.length; i++) {
			for (int j = i + 1; j < CHANNEL_ROLES.length; j++) {
				if (request.channels[i] != CHANNEL_ABSENT
					&& request.channels[i] == request.channels[j])
				{
					return "Channel " + request.channels[i] + " is assigned to both "
						+ CHANNEL_ROLES[i] + " and " + CHANNEL_ROLES[j]
						+ ".\n\nEach channel of the image can carry only one marker. "
						+ "Set the one the image does not have to 0.";
				}
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
		// One canonical string, built from the dialog's numbers rather than typed:
		// Python owns parsing and validating channel layouts, and this side owns
		// collecting them, so there is no second place a spelling could drift.
		inputs.put("channel_order", request.channelOrder());
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
			+ "  (" + request.describeChannels()
			+ "; cellprob " + request.cellprobThreshold
			+ ", flow " + request.flowThreshold
			+ (request.measures(ROLE_RFP)
				? ", RFP " + request.rfpMethod + " gate " + activeRfpGate(request) : "")
			+ ") ...");
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

		// What the worker actually measured, which can be narrower than what the
		// dialog declared: a marker needs its channel AND its checkpoint, and a
		// model directory may ship only one of the two. Null from an older
		// worker, in which case the dialog's declaration is the best available
		// answer.
		@SuppressWarnings("unchecked")
		final List<String> markers = (List<String>) task.outputs.get("markers");

		logBatch(task, images);
		renderResults(cells, request, markers == null ? request.markers() : markers);
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
			// Only the markers this image was actually scored for. A "RFP+ 0" for
			// an image with no RFP channel would read as a finding rather than as
			// an absent measurement -- which is the whole distinction the channel
			// settings introduce.
			@SuppressWarnings("unchecked")
			final List<String> markers = (List<String>) image.get("markers");
			final StringBuilder counts = new StringBuilder();
			counts.append(image.get("n_cells")).append(" cells");
			appendCount(counts, markers, ROLE_OPC, image.get("n_opc_pos"));
			appendCount(counts, markers, ROLE_B3TUB, image.get("n_b3tub_pos"));
			appendCount(counts, markers, ROLE_RFP, image.get("n_rfp_pos"));
			IJ.log("Neural Image Classifier: " + name
				+ " -- " + counts
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

	/** Appends {@code ", OPC+ 311"} when that marker was measured, else nothing.
	 *  A null {@code markers} means an older worker that did not report one, so
	 *  every marker is shown -- the historical behaviour. */
	private static void appendCount(
		final StringBuilder out, final List<String> markers,
		final String marker, final Object count
	) {
		if (markers != null && !markers.contains(marker)) return;
		out.append(", ").append(marker).append("+ ").append(count);
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
				// Named, never inferred. Appose picks a scheme by sniffing the
				// START of the content -- EnvironmentYmlScheme.supportsContent
				// asks whether it begins with "name:"/"channels:"/"dependencies:"
				// -- and environment.yml opens with a comment block explaining
				// what it is. Every scheme then declines and build() dies with
				// "Cannot infer scheme from content", before any environment is
				// created. We know what this file is, so say so rather than
				// making a documentation comment load-bearing.
				.scheme("environment.yml")
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
		final List<Map<String, Object>> cells, final ClassifyRequest request,
		final List<String> markers
	) {
		if (cells == null || cells.isEmpty()) {
			IJ.log("Neural Image Classifier: no cells to show.");
			return;
		}

		// `markers` is what the run measured, so the table carries columns only
		// for those. Every cell arrives with all of them -- Python keeps the row
		// shape fixed -- but a column of 0s for a channel that was never
		// acquired reads as "no cell was positive", which is a different claim.
		final boolean hasOpc = markers.contains(ROLE_OPC);
		final boolean hasB3Tub = markers.contains(ROLE_B3TUB);
		final boolean hasRfp = markers.contains(ROLE_RFP);
		if (markers.size() < MARKER_ROLES.length) {
			IJ.log("Neural Image Classifier: results table shows "
				+ (markers.isEmpty() ? "cell counts only" : String.join(", ", markers))
				+ " -- " + request.describeChannels());
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
			final boolean opcPos = hasOpc && (Boolean) c.get("opc_pos");
			final boolean b3tubPos = hasB3Tub && (Boolean) c.get("b3tub_pos");
			final boolean rfpPos = hasRfp && (Boolean) c.get("rfp_pos");

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
			if (hasOpc) {
				rt.addValue("opc_prob", ((Number) c.get("opc_prob")).doubleValue());
				rt.addValue("opc_pos", opcPos ? 1 : 0);
			}
			if (hasB3Tub) {
				rt.addValue("b3tub_prob", ((Number) c.get("b3tub_prob")).doubleValue());
				rt.addValue("b3tub_pos", b3tubPos ? 1 : 0);
			}
			if (hasRfp) {
				rt.addValue("rfp_pos", rfpPos ? 1 : 0);
				// Both RFP rules travel with every cell, whichever one made the
				// call, so the table can be re-read under the other one without
				// re-running the image. They are NOT on the same scale --
				// different background estimators -- so a single gate cannot be
				// compared across the two columns.
				rt.addValue("rfp_method", String.valueOf(c.get("rfp_method")));
				rt.addValue("rfp_neighbour_score",
					((Number) c.get("rfp_neighbour_score")).doubleValue());
				rt.addValue("rfp_neighbour_pos", (Boolean) c.get("rfp_neighbour_pos") ? 1 : 0);
				rt.addValue("rfp_legacy_pos", (Boolean) c.get("rfp_legacy_pos") ? 1 : 0);
				rt.addValue("rfp_score", ((Number) c.get("rfp_score")).doubleValue());
			}
			// The continuous channel scores are intensity measurements, not
			// classifier output, so they exist for any marker whose *channel* is
			// present -- including one whose checkpoint the model directory did
			// not ship.
			if (hasOpc) {
				rt.addValue("opc_channel_score",
					((Number) c.get("opc_channel_score")).doubleValue());
			}
			if (hasB3Tub) {
				rt.addValue("b3tub_channel_score",
					((Number) c.get("b3tub_channel_score")).doubleValue());
			}
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
		gd.addMessage("Select one ROI in the ROI Manager to pre-fill the id.",
			HINT_FONT, HINT_COLOR);
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
			stack.addSlice(
				channelName(channelNames, c), new ShortProcessor(w, h, plane, null));
		}

		final int cellId = ((Number) cell.get("cell_id")).intValue();
		// channelName(c) rather than channelNames.get(c): the stack loop above
		// already tolerates a short list, and the two must not disagree about
		// how many channels they think there are.
		final ImagePlus imp = new ImagePlus(
			roiName(String.valueOf(cell.get("image_name")), cellId), stack);
		imp.setDimensions(nChannels, 1, 1);
		final CompositeImage comp = new CompositeImage(imp, CompositeImage.COMPOSITE);
		for (int c = 0; c < nChannels; c++) {
			comp.setChannelLut(
				LUT.createLutFromColor(channelColor(channelName(channelNames, c))), c + 1);
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

	/** The worker's name for channel {@code index}, or a positional fallback.
	 *
	 *  Python sends one name per plane, so the fallback is only reached if that
	 *  ever stops being true -- in which case a composite with an unlabelled
	 *  channel is a far better outcome than an exception. */
	private static String channelName(final List<String> names, final int index) {
		return names != null && index < names.size() ? names.get(index) : ("ch" + (index + 1));
	}

	/** Cosmetic per-channel display color, by the role the run assigned that
	 *  channel. A channel the run does not use comes back named "ch<N>" and
	 *  falls through to grayscale. */
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
		gd.addMessage("Model bundle " + ModelBootstrap.BUNDLE_VERSION, SECTION_FONT);
		gd.addDirectoryField("Model directory:", DEFAULT_MODEL_DIR);
		addHint(gd, "OK fetches the checkpoints (~90 MB) unless they are already "
			+ "there; a complete directory is left untouched.");
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
