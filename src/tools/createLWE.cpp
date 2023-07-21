// =============================================================================
//  CADET
//  
//  Copyright © 2008-2022: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <cmath>

#include <tclap/CmdLine.h>
#include "common/TclapUtils.hpp"
#include "io/hdf5/HDF5Writer.hpp"
#include "ToolsHelper.hpp"

struct ProgramOptions
{
	std::string fileName;
	bool isKinetic;
	bool solverTimes;
	double startTime;
	double endTime;
	double constAlg;
	double stddevAlg;
	bool radialFlow;
	bool velocityDependence;
	bool reverseFlow;
	bool adJacobian;
	int nPar;
	int nCol;
	int nRad;
	int nThreads;
	std::vector<std::string> sensitivities;
	std::string outSol;
	std::string outSens;
	std::string unitType;
};


int main(int argc, char** argv)
{
	ProgramOptions opts;
	const double nanVal = std::numeric_limits<double>::quiet_NaN();

	try
	{
		TCLAP::CustomOutputWithoutVersion customOut("createLWE");
		TCLAP::CmdLine cmd("Create an HDF5 input file for load-wash-elution benchmark case", ' ', "1.0");
		cmd.setOutput(&customOut);

		cmd >> (new TCLAP::ValueArg<std::string>("o", "out", "Write output to file (default: LWE.h5)", false, "LWE.h5", "File"))->storeIn(&opts.fileName);
		cmd >> (new TCLAP::ValueArg<double>("t", "startTime", "Start time of simulation (default: 0sec)", false, 0.0, "Time"))->storeIn(&opts.startTime);
		cmd >> (new TCLAP::ValueArg<double>("T", "endTime", "End time of simulation (default: 1500sec)", false, 1500.0, "Time"))->storeIn(&opts.endTime);
		cmd >> (new TCLAP::ValueArg<double>("c", "constAlg", "Set all algebraic variables to constant value", false, nanVal, "Value"))->storeIn(&opts.constAlg);
		cmd >> (new TCLAP::ValueArg<double>("s", "stddevAlg", "Perturb algebraic variables with normal variates", false, nanVal, "Value"))->storeIn(&opts.stddevAlg);
		cmd >> (new TCLAP::SwitchArg("", "reverseFlow", "Reverse the flow for column"))->storeIn(&opts.reverseFlow);
		cmd >> (new TCLAP::SwitchArg("", "radialFlow", "Use radial flow column"))->storeIn(&opts.radialFlow);
		cmd >> (new TCLAP::SwitchArg("", "velDep", "Use velocity dependent dispersion and film diffusion"))->storeIn(&opts.velocityDependence);
		cmd >> (new TCLAP::ValueArg<int>("", "rad", "Number of radial cells (default: 3)", false, 3, "Value"))->storeIn(&opts.nRad);
		addMiscToCmdLine(cmd, opts);
		addUnitTypeToCmdLine(cmd, opts.unitType);
		addSensitivitiyParserToCmdLine(cmd, opts.sensitivities);
		addOutputParserToCmdLine(cmd, opts.outSol, opts.outSens);

		cmd.parse(argc, argv);
	}
	catch (const TCLAP::ArgException &e)
	{
		std::cerr << "ERROR: " << e.error() << " for argument " << e.argId() << std::endl;
		return 1;
	}

	cadet::io::HDF5Writer writer;
	writer.openFile(opts.fileName, "co");
	writer.pushGroup("input");

	parseUnitType(opts.unitType, opts.radialFlow);
	const bool isGRM2D = (opts.unitType == "GENERAL_RATE_MODEL_2D");

	// Model
	{
		Scope<cadet::io::HDF5Writer> s(writer, "model");
		writer.scalar<int>("NUNITS", 2);

		{
			Scope<cadet::io::HDF5Writer> su(writer, "unit_000");

			writer.scalar("UNIT_TYPE", opts.unitType);
			const int nComp = 4;
			writer.scalar<int>("NCOMP", nComp);

			// Transport
			if (!opts.reverseFlow)
				writer.scalar<double>("VELOCITY", 1.0);
			else
				writer.scalar<double>("VELOCITY", -1.0);
			writer.scalar<double>("COL_DISPERSION", 5.75e-8);
			writer.scalar<double>("COL_DISPERSION_RADIAL", 1e-6);

			const double filmDiff[] = {6.9e-6, 6.9e-6, 6.9e-6, 6.9e-6};
			const double parDiff[] = {7e-10, 6.07e-11, 6.07e-11, 6.07e-11};
			const double parSurfDiff[] = {0.0, 0.0, 0.0, 0.0};

			writer.vector<double>("FILM_DIFFUSION", 4, filmDiff);
			writer.vector<double>("PAR_DIFFUSION", 4, parDiff);
			writer.vector<double>("PAR_SURFDIFFUSION", 4, parSurfDiff);

			if (opts.velocityDependence)
			{
				writer.scalar<std::string>("COL_DISPERSION_DEP", "POWER_LAW");
				writer.scalar<double>("COL_DISPERSION_DEP_BASE", 1.25);
				writer.scalar<double>("COL_DISPERSION_DEP_EXPONENT", 1.0);

				writer.scalar<std::string>("FILM_DIFFUSION_DEP", "POWER_LAW");
				writer.scalar<double>("FILM_DIFFUSION_DEP_BASE", 1.25);
				writer.scalar<double>("FILM_DIFFUSION_DEP_EXPONENT", 1.0);
			}

			// Geometry
			if (opts.radialFlow)
				writer.scalar<double>("COL_LENGTH", 0.0014);
			else
				writer.scalar<double>("COL_LENGTH", 0.014);
			writer.scalar<double>("COL_RADIUS", 0.01);
			writer.scalar<double>("COL_RADIUS_INNER", 0.01);
			writer.scalar<double>("COL_RADIUS_OUTER", 0.04);
			writer.scalar<double>("CROSS_SECTION_AREA", 0.0003141592653589793);
			writer.scalar<double>("PAR_RADIUS", 4.5e-5);
			writer.scalar<double>("PAR_CORERADIUS", 0.0);
			writer.scalar<double>("COL_POROSITY", 0.37);
			writer.scalar<double>("PAR_POROSITY", 0.75);
			writer.scalar<double>("TOTAL_POROSITY", 0.37 + (1.0 - 0.37) * 0.75);

			// Initial conditions
			const double initC[] = {50.0, 0.0, 0.0, 0.0};
			const double initQ[] = {1.2e3, 0.0, 0.0, 0.0};
			writer.vector<double>("INIT_C", 4, initC);
			writer.vector<double>("INIT_Q", 4, initQ);

/*
			double stateY[] = {2.4091509252666779e+02, 2.4024398465113154e+02, 2.3932473703038917e+02, 2.3831769254224670e+02, 2.3729889106920447e+02, 2.3629503580553160e+02, 2.3527793331631426e+02, 2.3429522482544124e+02, 2.3323145402824704e+02, 2.3226971398887699e+02, 2.3081194362901846e+02, 5.4238186038543688e-03, 8.7492844964037440e-03, 1.2899996998026011e-02, 1.6059433635769331e-02, 1.7508477029228128e-02, 1.7184483461874935e-02, 1.5495262336194349e-02, 1.3145606411698525e-02, 1.0371513902339438e-02, 8.0568259053558598e-03, 5.1769973885380394e-03, 1.2528741741403339e-05, 3.0471625429747610e-05, 7.1597882107318734e-05, 1.7284887571502813e-04, 4.0272385451813568e-04, 8.7075445165527021e-04, 1.7567861082580982e-03, 3.2109063605366985e-03, 5.5051863714491670e-03, 8.7810378220644809e-03, 1.4768343845033009e-02, -1.0452358828165517e-11, -3.4326093024519636e-10, 1.8302667938408549e-09, -3.7404812117352811e-09, 8.4284080659989416e-09, -1.5874327910430133e-08, 3.7974018630824683e-08, -5.2468469149304292e-08, 1.6626175191659887e-07, -7.5172853122698841e-08, 9.1708954950005931e-07, 2.4055840761533423e+02, 1.0063769557000243e-02, 2.1722719356323400e-05, 5.7095132095810761e-13, 1.1969297505989468e+03, 6.5306211539281978e-01, 1.6209049166526499e-04, 1.6256238351228343e-12, 2.4053031707427616e+02, 1.3285126496152481e-02, 2.8122113236550063e-05, 7.7178554604213204e-12, 1.1959970822005789e+03, 8.5145173689426090e-01, 2.0692550808163186e-04, 2.1760405004647323e-11, 2.4050773359697263e+02, 1.6139667379799185e-02, 3.3834331703802583e-05, 1.3756513083022610e-11, 1.1951893819802144e+03, 1.0232589350876868e+00, 2.4594040286873338e-04, 3.8457056549710381e-11, 2.4049073143612722e+02, 1.8441825605610036e-02, 3.8482321077057996e-05, 1.8473509089080953e-11, 1.1945503025712512e+03, 1.1591982847484978e+00, 2.7703029128079023e-04, 5.1295003651058555e-11, 2.4047936630035909e+02, 2.0052418000605979e-02, 4.1760343956465238e-05, 2.1707842659949091e-11, 1.1941095288913446e+03, 1.2529556174572429e+00, 2.9862124471381415e-04, 5.9993945692768222e-11, 2.4047367494229266e+02, 2.0879659408940369e-02, 4.3453054523323253e-05, 2.3351321150663286e-11, 1.1938851142631540e+03, 1.3006909796847199e+00, 3.0966580148476824e-04, 6.4382000842918285e-11, 2.3988736237616479e+02, 1.3334085513301244e-02, 5.2123133708533173e-05, -2.4552639035224615e-10, 1.1959340869476084e+03, 8.6465048733668648e-01, 3.8861332593050925e-04, -6.9866668191312299e-10, 2.3985934798161071e+02, 1.6339454312764023e-02, 6.7152563355529459e-05, -1.8235588837675413e-10, 1.1950742787781446e+03, 1.0474694914381208e+00, 4.9425595445185119e-04, -5.1425239264238592e-10, 2.3983687882940555e+02, 1.8884243836178788e-02, 8.0548147408277982e-05, -1.2932723477593460e-10, 1.1943610523070338e+03, 1.1991159584821844e+00, 5.8651974283088456e-04, -3.6198085405553739e-10, 2.3981999667122105e+02, 2.0866200649876925e-02, 9.1440355922715541e-05, -8.8209968645812170e-11, 1.1938146305637677e+03, 1.3152927904400711e+00, 6.6036334070075685e-04, -2.4547534883710258e-10, 2.3980872891035114e+02, 2.2218737527169345e-02, 9.9120147951821231e-05, -6.0198830342385429e-11, 1.1934461607583899e+03, 1.3936326742489664e+00, 7.1184749673849869e-04, -1.6687282735446475e-10, 2.3980309152618980e+02, 2.2903378936807817e-02, 1.0308561102248533e-04, -4.6025528567556974e-11, 1.1932609850890963e+03, 1.4330020318487957e+00, 7.3825362677692053e-04, -1.2733412546368171e-10, 2.3897054031814505e+02, 1.6652693831085925e-02, 1.2112789223649425e-04, 1.5895454294541986e-09, 1.1949016842844676e+03, 1.0837274505976178e+00, 9.0674497926880506e-04, 4.5359836887157740e-09, 2.3894281600355026e+02, 1.8898551704049815e-02, 1.5553831429835924e-04, 1.4225417404396116e-09, 1.1942618431192509e+03, 1.2195862854503943e+00, 1.1533694161655777e-03, 4.0326350393868773e-09, 2.3892064996970564e+02, 2.0653509586353994e-02, 1.8625154695485897e-04, 1.2753148565743870e-09, 1.1937688263827756e+03, 1.3242384111519667e+00, 1.3710910211039522e-03, 3.5968958527381765e-09, 2.3890403920767625e+02, 2.1930089342053861e-02, 2.1126431315602451e-04, 1.1572323932248313e-09, 1.1934138853125537e+03, 1.3995596878703711e+00, 1.5470968664375601e-03, 3.2519195473964295e-09, 2.3889297429291986e+02, 2.2756158048436722e-02, 2.2892437816440399e-04, 1.0749815417057558e-09, 1.1931858011030592e+03, 1.4479490137463362e+00, 1.6707979550912920e-03, 3.0136958129199627e-09, 2.3888744493784515e+02, 2.3160618544673207e-02, 2.3805139931716703e-04, 1.0328460631049647e-09, 1.1930745722690394e+03, 1.4715429458233114e+00, 1.7345699224071711e-03, 2.8922566995908284e-09, 2.3796781166111577e+02, 1.8348760793381248e-02, 2.8839848599951753e-04, -3.3472260885844950e-09, 1.1943111599040803e+03, 1.2079299099182910e+00, 2.1870614304607496e-03, -9.6387185815307336e-09, 2.3794050587085405e+02, 1.9502071883939830e-02, 3.6865061960993724e-04, -3.0481722928853998e-09, 1.1939761514551790e+03, 1.2785379771452190e+00, 2.7826246898825882e-03, -8.7489327137832604e-09, 2.3791873167532631e+02, 2.0235523137803031e-02, 4.4031896816293246e-04, -2.7677254002339888e-09, 1.1937632277403222e+03, 1.3232427485537410e+00, 3.3140587540825927e-03, -7.9280516711321806e-09, 2.3790245029301363e+02, 2.0654195577124970e-02, 4.9873930663018630e-04, -2.5329623246924917e-09, 1.1936410623274855e+03, 1.3487470880131491e+00, 3.7478989922319858e-03, -7.2476567623195917e-09, 2.3789162272404857e+02, 2.0862435099409028e-02, 5.4002495127271799e-04, -2.3647126916311150e-09, 1.1935796520081562e+03, 1.3614672292713936e+00, 4.0552059150226272e-03, -6.7628041691154569e-09, 2.3788621742061449e+02, 2.0944086955397741e-02, 5.6137599070953754e-04, -2.2771282969355968e-09, 1.1935552675091135e+03, 1.3664762110082134e+00, 4.2144278334367638e-03, -6.5111242895838227e-09, 2.3695378529143363e+02, 1.8136652521364295e-02, 6.5599741057026438e-04, 8.3279497219621472e-09, 1.1942522428914845e+03, 1.2172049910779108e+00, 5.0838488396740015e-03, 2.4347969695228709e-08, 2.3692689587740452e+02, 1.8171023354073629e-02, 8.3122053827311274e-04, 8.1832393369984717e-09, 1.1942326398165014e+03, 1.2198452171641567e+00, 6.4437758574768791e-03, 2.3930038350477526e-08, 2.3690547937728513e+02, 1.7941634751691436e-02, 9.8731244574424117e-04, 8.0120564810858079e-09, 1.1942914115379745e+03, 1.2059664020915184e+00, 7.6647042382112828e-03, 2.3452731666128873e-08, 2.3688947873240724e+02, 1.7603326546064869e-02, 1.1143530285842120e-03, 7.8513101540296640e-09, 1.1943839463003858e+03, 1.1851502900097630e+00, 8.6667772070796092e-03, 2.3011606958364772e-08, 2.3687884348713254e+02, 1.7292223694402607e-02, 1.2040503176555975e-03, 7.7287912997473366e-09, 1.1944707632888224e+03, 1.1658763749689487e+00, 9.3795207778219064e-03, 2.2678108106108440e-08, 2.3687353570371781e+02, 1.7110285033995004e-02, 1.2504160755739391e-03, 7.6630474487928605e-09, 1.1945219758736391e+03, 1.1545633504668202e+00, 9.7497723851656393e-03, 2.2499831584664851e-08, 2.3595363654662478e+02, 1.6382113549674331e-02, 1.3713342640815931e-03, -1.4864002690469598e-08, 1.1946417989642589e+03, 1.1277342682784623e+00, 1.0935754039862518e-02, -4.4334705135505447e-08, 2.3592702782857211e+02, 1.5562373448831662e-02, 1.7152528026343674e-03, -1.3912157305695565e-08, 1.1948730565654330e+03, 1.0753777387239378e+00, 1.3736902940994281e-02, -4.1619817985042192e-08, 2.3590582470065809e+02, 1.4668005924478299e-02, 2.0200350582753931e-03, -1.2910870077880058e-08, 1.1951308714706352e+03, 1.0176929215638415e+00, 1.6251784760212773e-02, -3.8747805241235187e-08, 2.3588997463164179e+02, 1.3848644212653600e-02, 2.2671511956411229e-03, -1.2012332288023409e-08, 1.1953704706665742e+03, 9.6439134645417635e-01, 1.8315716232105893e-02, -3.6155868660903321e-08, 2.3587943399795867e+02, 1.3228322762887809e-02, 2.4411729184424542e-03, -1.1340372490462095e-08, 1.1955536690736731e+03, 9.2376095625791421e-01, 1.9783470413056961e-02, -3.4208211575338827e-08, 2.3587417155110936e+02, 1.2895812663903218e-02, 2.5309926245012581e-03, -1.0982508371478752e-08, 1.1956524519299669e+03, 9.0188525630522365e-01, 2.0545838963797455e-02, -3.3167636521299849e-08, 2.3493818146223489e+02, 1.3762229043992611e-02, 2.6571219067982278e-03, 4.2855382370581485e-08, 1.1952978255718890e+03, 9.7580644368192015e-01, 2.1906173671264170e-02, 1.3083380650490700e-07, 2.3491166521777376e+02, 1.2450991049539919e-02, 3.2687564007921953e-03, 4.5706470501543663e-08, 1.1956823829308801e+03, 8.8810624046298137e-01, 2.7129909305804344e-02, 1.4019358936246558e-07, 2.3489050485264318e+02, 1.1227172804320189e-02, 3.8059638894526789e-03, 4.7922490682752470e-08, 1.1960470457793460e+03, 8.0527687656847224e-01, 3.1786834505048663e-02, 1.4763525974850708e-07, 2.3487466459603675e+02, 1.0207655902942045e-02, 4.2385457826447240e-03, 4.9547256595788717e-08, 1.1963546315154240e+03, 7.3555798714974818e-01, 3.5585137643448755e-02, 1.5319953415773654e-07, 2.3486411827780401e+02, 9.4801846181688347e-03, 4.5416867947768481e-03, 5.0613260826430149e-08, 1.1965761068222812e+03, 6.8541025530243482e-01, 3.8273042957882239e-02, 1.5690546033665456e-07, 2.3485884906871723e+02, 9.1026216271528973e-03, 4.6977019605757781e-03, 5.1141378664401615e-08, 1.1966916919038190e+03, 6.5925113236760746e-01, 3.9664874602421504e-02, 1.5875852713825124e-07, 2.3395517179669946e+02, 1.0944880891303185e-02, 4.6477129660580686e-03, -4.8005369453883916e-08, 1.1960313344090093e+03, 7.9978208610555002e-01, 3.9639005967693972e-02, -1.5007417106810249e-07, 2.3392858020319980e+02, 9.4561131311612871e-03, 5.6122418742291375e-03, -4.3416372299547480e-08, 1.1964750050898213e+03, 6.9571097595602382e-01, 4.8233237821728225e-02, -1.3645717916763455e-07, 2.3390732000971138e+02, 8.1724743727023281e-03, 6.4512916058415935e-03, -3.8364021615166865e-08, 1.1968620771497538e+03, 6.0482370692908860e-01, 5.5813209053172427e-02, -1.2113832073680170e-07, 2.3389137896943507e+02, 7.1628801047047778e-03, 7.1218880726488711e-03, -3.3700754923206467e-08, 1.1971692248753338e+03, 5.3257897149794287e-01, 6.1938441172245458e-02, -1.0680414949588045e-07, 2.3388075209353940e+02, 6.4706112521194497e-03, 7.5893035448222259e-03, -3.0151694757347033e-08, 1.1973811272607541e+03, 4.8264978907713912e-01, 6.6241793012229530e-02, -9.5797650303893752e-08, 2.3387543849005729e+02, 6.1195093807209014e-03, 7.8291047470789212e-03, -2.8243586620149182e-08, 1.1974889867592990e+03, 4.5720414559043382e-01, 6.8460130221528567e-02, -8.9850261910725981e-08, 2.3288910296216710e+02, 8.1365896042057701e-03, 7.6518055502512615e-03, 2.1854474508882763e-07, 1.1967571875504848e+03, 6.1382461149658385e-01, 6.7643512923426213e-02, 7.0057205697447517e-07, 2.3286227926914228e+02, 6.7341253193051323e-03, 9.0705603090421583e-03, 2.5225749725209578e-07, 1.1971696709868793e+03, 5.1129166551320859e-01, 8.0766574307323535e-02, 8.1273595012573467e-07, 2.3284079745996152e+02, 5.5943014806029640e-03, 1.0289066678020813e-02, 2.8068415165236007e-07, 1.1975057162511412e+03, 4.2697305655271950e-01, 9.2156337105749747e-02, 9.0804632549207547e-07, 2.3282466870113905e+02, 4.7377926315738126e-03, 1.1253237418996989e-02, 3.0288654790981646e-07, 1.1977582332511035e+03, 3.6302384901892665e-01, 1.0123837840367647e-01, 9.8290590461534124e-07, 2.3281390636839689e+02, 4.1694136387099819e-03, 1.1920464489358702e-02, 3.1812103048199253e-07, 1.1979255875802355e+03, 3.2030603164955662e-01, 1.0755581162579833e-01, 1.0344624259001074e-06, 2.3280852201820241e+02, 3.8866739069951628e-03, 1.2261332297133316e-02, 3.2586778917796842e-07, 1.1980087330795584e+03, 2.9897215968424079e-01, 1.1079278737409731e-01, 1.0607342722767905e-06, 2.3192437625639178e+02, 6.0090234520040735e-03, 1.1609473384929318e-02, -2.7065361454992854e-08, 1.1972519976106967e+03, 4.6563134041685311e-01, 1.0577229058686216e-01, -8.8610841131218960e-08, 2.3189728772288493e+02, 4.7862703883071100e-03, 1.3450116040944118e-02, 1.2728285469445827e-08, 1.1975949782491209e+03, 3.7292580183953628e-01, 1.2330251935322370e-01, 4.1852555606799969e-08, 2.3187557423893227e+02, 3.8336533790050312e-03, 1.5011423094221005e-02, 5.2593669445426369e-08, 1.1978586314113247e+03, 2.9997575975624258e-01, 1.3827634703408870e-01, 1.7351636739544536e-07, 2.3185926068527635e+02, 3.1421728691755856e-03, 1.6235092805474093e-02, 8.7677416195715365e-08, 1.1980470218501355e+03, 2.4662393246029835e-01, 1.5006514072525037e-01, 2.8996339487913059e-07, 2.3184837017957287e+02, 2.6950601880010869e-03, 1.7076194654562366e-02, 1.1371776345262688e-07, 1.1981670196108673e+03, 2.1194776213127289e-01, 1.5818988912090279e-01, 3.7666650558166824e-07, 2.3184292035004000e+02, 2.4761299631719550e-03, 1.7504204008952653e-02, 1.2755027553246449e-07, 1.1982251605901847e+03, 1.9491736974822335e-01, 1.6233009595974227e-01, 4.2280310345648748e-07, 2.3046182949752750e+02, 3.6238865614504705e-03, 1.8487708380326314e-02, 1.5180536234704331e-06, 1.1977021567949573e+03, 2.9131469910358671e-01, 1.7554729502526450e-01, 5.1158302938563231e-06, 2.3043434845170361e+02, 2.7549752411379167e-03, 2.0853389567577093e-02, 1.9339147307452061e-06, 1.1979036056377886e+03, 2.2226417194125153e-01, 1.9881447480063469e-01, 6.5357817076906969e-06, 2.3041231285934316e+02, 2.1140696484413008e-03, 2.2822977682319643e-02, 2.3040433494295189e-06, 1.1980418848759653e+03, 1.7099687415096845e-01, 2.1822324130555773e-01, 7.8024374690915028e-06, 2.3039575522320402e+02, 1.6688665416119803e-03, 2.4344467089754282e-02, 2.6048488015544385e-06, 1.1981306782437562e+03, 1.3521993603586818e-01, 2.3322407834270434e-01, 8.8330926058812407e-06, 2.3038470167930245e+02, 1.3901825686677216e-03, 2.5379573394930802e-02, 2.8169187693749314e-06, 1.1981822705423374e+03, 1.1275774839046174e-01, 2.4342772548233801e-01, 9.5601151586103087e-06, 2.3037917041276040e+02, 1.2563645969664347e-03, 2.5903150368092424e-02, 2.9264256991878912e-06, 1.1982057515225401e+03, 1.0195401583070886e-01, 2.4858749362434926e-01, 9.9356169889488713e-06, 2.3455250125819052e-06, 2.3451131219748598e-06, 2.3291628592662535e-06, 2.3007823769680198e-06, 2.2693817653647426e-06, 2.2450080605024685e-06, 2.2341748879071539e-06, 2.2361553831495584e-06, 2.2512670512419287e-06, 2.2709070769013426e-06, 2.3023162084065774e-06, -2.0413347244816752e-08, -2.0170716490671570e-08, -1.6509895109611741e-08, -1.0071837122382583e-08, -2.7636422314778409e-09, 3.5300062094398306e-09, 7.6244362735797828e-09, 9.6820364408777673e-09, 9.8324931053412855e-09, 9.0092552679795077e-09, 6.8328719296786543e-09, -4.0448672822887483e-11, -9.5255265034223011e-11, -2.1790603129751930e-10, -5.0835759794484531e-10, -1.1142706260659880e-09, -2.2022882640752239e-09, -3.9610046328978296e-09, -6.3211944678752580e-09, -9.4439969389881597e-09, -1.2443630869458432e-08, -1.6363250529139321e-08, -4.8496783243779168e-17, -4.2998099394247235e-16, 1.0590497421765096e-15, -1.7301191568476195e-15, 4.4196140605152627e-16, -4.4448714812807004e-15, -2.1475515369311758e-14, -1.9635179056790156e-14, -2.3001788388642278e-13, -2.1164792260310935e-13, -2.6439260081154732e-12};

			if (!isKinetic && (!std::isnan(stddevAlg) || !std::isnan(constAlg)))
			{
				// Initialize standard normal RNG
				unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
				std::default_random_engine generator(seed);
				std::normal_distribution<double> distribution(0.0, 1);

				// Jump over column
				const int offset = 11 * nComp;
				const int strideShell = nComp * 2;
				const int stridePar = strideShell * 6;
				const int offsetFlux = 11 * nComp + stridePar * 11;

				for (int par = 0; par < 11; ++par)
				{
					for (int shell = 0; shell < 6; ++shell)
					{
						const int localOffset = offset + par * stridePar + shell * strideShell + nComp;

						if (!std::isnan(stddevAlg))
						{
							for (int i = 0; i < nComp; ++i)
								stateY[localOffset + i] += stddevAlg * stateY[localOffset + i] * distribution(generator);
						}
						else if (!std::isnan(constAlg))
						{
							for (int i = 0; i < nComp; ++i)
								stateY[localOffset + i] = constAlg;
						}
					}

					// Fluxes
					if (!std::isnan(stddevAlg))
					{
						for (int i = 0; i < nComp; ++i)
							stateY[offsetFlux + i] += stddevAlg * stateY[offsetFlux + i] * distribution(generator);
					}
					else if (!std::isnan(constAlg))
					{
						for (int i = 0; i < nComp; ++i)
							stateY[offsetFlux + i] = constAlg;
					}
				}

				writer.vector<double>("INIT_STATE", offsetFlux + 11 * nComp, stateY);
			}
*/

			// Adsorption
			writer.scalar("ADSORPTION_MODEL", std::string("STERIC_MASS_ACTION"));
			{
				Scope<cadet::io::HDF5Writer> s2(writer, "adsorption");
				
				writer.scalar<int>("IS_KINETIC", opts.isKinetic);
		
				const double kA[] = {0.0, 35.5, 1.59, 7.7};
				const double kD[] = {0.0, 1000.0, 1000.0, 1000.0};
				writer.vector<double>("SMA_KA", 4, kA);
				writer.vector<double>("SMA_KD", 4, kD);

				writer.scalar<double>("SMA_LAMBDA", 1.2e3);

				const double nu[] = {0.0, 4.7, 5.29, 3.7};
				const double sigma[] = {0.0, 11.83, 10.6, 10.0};
				writer.vector<double>("SMA_NU", 4, nu);
				writer.vector<double>("SMA_SIGMA", 4, sigma);
			}

			// Discretization
			{
				Scope<cadet::io::HDF5Writer> s2(writer, "discretization");

				writer.scalar<int>("NCOL", opts.nCol); // 64
				writer.scalar<int>("NPAR", opts.nPar); // 16
				writer.scalar<int>("NRAD", opts.nRad);
				const int nBound[] = {1, 1, 1, 1};
				writer.vector<int>("NBOUND", 4, nBound);

				writer.scalar("RADIAL_DISC_TYPE", std::string("EQUIDISTANT"));
				writer.scalar("PAR_DISC_TYPE", std::string("EQUIDISTANT_PAR"));

				writer.scalar<int>("USE_ANALYTIC_JACOBIAN", !opts.adJacobian);
				writer.scalar<int>("MAX_KRYLOV", 0);
				writer.scalar<int>("GS_TYPE", 1);
				writer.scalar<int>("MAX_RESTARTS", 10);
				writer.scalar<double>("SCHUR_SAFETY", 1e-8);

				// WENO
				{
					Scope<cadet::io::HDF5Writer> s3(writer, "weno");

					writer.scalar<int>("WENO_ORDER", 3);
					writer.scalar<int>("BOUNDARY_MODEL", 0);
					writer.scalar<double>("WENO_EPS", 1e-10);
				}
			}
		}

		// Inlet - unit 001
		{
			Scope<cadet::io::HDF5Writer> su(writer, "unit_001");

			writer.scalar("UNIT_TYPE", std::string("INLET"));
			writer.scalar("INLET_TYPE", std::string("PIECEWISE_CUBIC_POLY"));
			writer.scalar<int>("NCOMP", 4);

			if (opts.startTime < 10.0)
			{
				{
					Scope<cadet::io::HDF5Writer> s3(writer, "sec_000");

					const double constCoeff[] = {50.0, 1.0, 1.0, 1.0};
					const double linCoeff[] = {0.0, 0.0, 0.0, 0.0};

					writer.vector<double>("CONST_COEFF", 4, constCoeff);
					writer.vector<double>("LIN_COEFF", 4, linCoeff);
					writer.vector<double>("QUAD_COEFF", 4, linCoeff);
					writer.vector<double>("CUBE_COEFF", 4, linCoeff);
				}

				{
					Scope<cadet::io::HDF5Writer> s3(writer, "sec_001");

					const double constCoeff[] = {50.0, 0.0, 0.0, 0.0};
					const double linCoeff[] = {0.0, 0.0, 0.0, 0.0};

					writer.vector<double>("CONST_COEFF", 4, constCoeff);
					writer.vector<double>("LIN_COEFF", 4, linCoeff);
					writer.vector<double>("QUAD_COEFF", 4, linCoeff);
					writer.vector<double>("CUBE_COEFF", 4, linCoeff);
				}

				{
					Scope<cadet::io::HDF5Writer> s3(writer, "sec_002");

					const double constCoeff[] = {100.0, 0.0, 0.0, 0.0};
					const double linCoeff[] = {0.2, 0.0, 0.0, 0.0};
					const double quadCoeff[] = {0.0, 0.0, 0.0, 0.0};

					writer.vector<double>("CONST_COEFF", 4, constCoeff);
					writer.vector<double>("LIN_COEFF", 4, linCoeff);
					writer.vector<double>("QUAD_COEFF", 4, quadCoeff);
					writer.vector<double>("CUBE_COEFF", 4, quadCoeff);
				}
			}
			else if (opts.startTime < 90.0)
			{
				{
					Scope<cadet::io::HDF5Writer> s3(writer, "sec_000");

					const double constCoeff[] = {50.0, 0.0, 0.0, 0.0};
					const double linCoeff[] = {0.0, 0.0, 0.0, 0.0};

					writer.vector<double>("CONST_COEFF", 4, constCoeff);
					writer.vector<double>("LIN_COEFF", 4, linCoeff);
					writer.vector<double>("QUAD_COEFF", 4, linCoeff);
					writer.vector<double>("CUBE_COEFF", 4, linCoeff);
				}

				{
					Scope<cadet::io::HDF5Writer> s3(writer, "sec_001");

					const double constCoeff[] = {100.0, 0.0, 0.0, 0.0};
					const double linCoeff[] = {0.2, 0.0, 0.0, 0.0};
					const double quadCoeff[] = {0.0, 0.0, 0.0, 0.0};

					writer.vector<double>("CONST_COEFF", 4, constCoeff);
					writer.vector<double>("LIN_COEFF", 4, linCoeff);
					writer.vector<double>("QUAD_COEFF", 4, quadCoeff);
					writer.vector<double>("CUBE_COEFF", 4, quadCoeff);
				}
			}
			else if (opts.startTime < 1500.0)
			{
				{
					Scope<cadet::io::HDF5Writer> s3(writer, "sec_000");

					const double constCoeff[] = {100.0 + (opts.startTime - 90.0) * 0.2, 0.0, 0.0, 0.0};
					const double linCoeff[] = {0.2, 0.0, 0.0, 0.0};
					const double quadCoeff[] = {0.0, 0.0, 0.0, 0.0};

					writer.vector<double>("CONST_COEFF", 4, constCoeff);
					writer.vector<double>("LIN_COEFF", 4, linCoeff);
					writer.vector<double>("QUAD_COEFF", 4, quadCoeff);
					writer.vector<double>("CUBE_COEFF", 4, quadCoeff);
				}
			}
		}

		// Valve switches
		{
			Scope<cadet::io::HDF5Writer> su(writer, "connections");
			writer.scalar<int>("NSWITCHES", 1);
			writer.scalar<int>("CONNECTIONS_INCLUDE_PORTS", 1);

			{
				Scope<cadet::io::HDF5Writer> s1(writer, "switch_000");

				if (!isGRM2D)
				{
					// Connection list is 1x7 since we have 1 connection between
					// the two unit operations (and we need to have 7 columns)
					const double connMatrix[] = {1, 0, -1, -1, -1, -1, 6.683738370512285e-8};
					// Connections: From unit operation 1 port -1 (i.e., all ports)
					//              to unit operation 0 port -1 (i.e., all ports),
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              a flow rate of 6.683738370512285e-8 m^3/s

					writer.vector<double>("CONNECTIONS", 7, connMatrix);
				}
				else
				{
					// Connection list is 3x7 since we have 1 connection between
					// the two unit operations with 3 ports (and we need to have 7 columns)
					const double connMatrix[] = {1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 7.42637597e-09,
					                             1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 2.22791279e-08,
					                             1.0, 0.0, 0.0, 2.0, -1.0, -1.0, 3.71318798e-08};
					// Connections: From unit operation 1 port 0
					//              to unit operation 0 port 0,
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 7.42637597e-09 m^3/s

					writer.vector<double>("CONNECTIONS", 21, connMatrix);
				}

				// This switch occurs at beginning of section 0 (initial configuration)
				writer.scalar<int>("SECTION", 0);
			}
		}

		// Solver settings
		{
			Scope<cadet::io::HDF5Writer> su(writer, "solver");

			writer.scalar<int>("MAX_KRYLOV", 0);
			writer.scalar<int>("GS_TYPE", 1);
			writer.scalar<int>("MAX_RESTARTS", 10);
			writer.scalar<double>("SCHUR_SAFETY", 1e-8);
		}
	}

	// Return
	{
		Scope<cadet::io::HDF5Writer> s(writer, "return");
		writer.template scalar<int>("WRITE_SOLUTION_TIMES", true);
	
		Scope<cadet::io::HDF5Writer> s2(writer, "unit_000");
		parseAndWriteOutputFormatsFromCmdLine(writer, opts.outSol, opts.outSens);
	}

	// Solver
	{
		Scope<cadet::io::HDF5Writer> s(writer, "solver");

		if (!opts.solverTimes)
		{
			std::vector<double> solTimes;
			solTimes.reserve(1501);
			for (double t = 0.0; t <= opts.endTime - opts.startTime; t += 1.0)
				solTimes.push_back(t);

			writer.vector<double>("USER_SOLUTION_TIMES", solTimes.size(), solTimes.data());
		}

		writer.scalar<int>("NTHREADS", opts.nThreads);

		// Sections
		{
			Scope<cadet::io::HDF5Writer> s2(writer, "sections");

			if (opts.startTime < 10.0)
			{
				writer.scalar<int>("NSEC", 3);

				const double secTimes[] = {0.0, 10.0 - opts.startTime, 90.0 - opts.startTime, 1500.0 - opts.startTime};
				writer.vector<double>("SECTION_TIMES", 4, secTimes);

				const int secCont[] = {0, 0};
				writer.vector<int>("SECTION_CONTINUITY", 2, secCont);
			}
			else if (opts.startTime < 90.0)
			{
				writer.scalar<int>("NSEC", 2);

				const double secTimes[] = {0.0, 90.0 - opts.startTime, 1500.0 - opts.startTime};
				writer.vector<double>("SECTION_TIMES", 3, secTimes);

				const int secCont[] = {0, 0};
				writer.vector<int>("SECTION_CONTINUITY", 1, secCont);
			}
			else if (opts.startTime < 1500.0)
			{
				writer.scalar<int>("NSEC", 1);

				const double secTimes[] = {0.0, 1500.0 - opts.startTime};
				writer.vector<double>("SECTION_TIMES", 2, secTimes);
			}
		}

		// Time integrator
		{
			Scope<cadet::io::HDF5Writer> s2(writer, "time_integrator");

			writer.scalar<double>("ABSTOL", 1e-8);
			writer.scalar<double>("RELTOL", 1e-6);
			writer.scalar<double>("ALGTOL", 1e-12);
			writer.scalar<double>("INIT_STEP_SIZE", 1e-6);
			writer.scalar<int>("MAX_STEPS", 10000);
		}
	}

	parseAndWriteSensitivitiesFromCmdLine(writer, opts.sensitivities);

	writer.closeFile();
	return 0;
}
