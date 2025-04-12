
// MatchToolDlg.h: 標頭檔
//
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <bits/fs_fwd.h>
using namespace cv;
using namespace std;
#pragma once

#define BOOL uint8_t
#define FALSE 0
#define TRUE 1

class Timer {
	private:
		std::string name;
		std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
	
	public:
		Timer(const std::string& timer_name = "Timer") : name(timer_name) {
			start_time = std::chrono::high_resolution_clock::now();
		}
	
		~Timer() {
			stop();
		}
	
		void stop() {
			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
			std::cout << name << ": " << duration.count() << "s" << std::endl;
		}
};


struct s_TemplData
{
	vector<Mat> vecPyramid;
	vector<Scalar> vecTemplMean;
	vector<double> vecTemplNorm;
	vector<double> vecInvArea;
	vector<BOOL> vecResultEqual1;
	BOOL bIsPatternLearned;
	int iBorderColor;
	void clear ()
	{
		vector<Mat> ().swap (vecPyramid);
		vector<double> ().swap (vecTemplNorm);
		vector<double> ().swap (vecInvArea);
		vector<Scalar> ().swap (vecTemplMean);
		vector<BOOL> ().swap (vecResultEqual1);
	}
	void resize (int iSize)
	{
		vecTemplMean.resize (iSize);
		vecTemplNorm.resize (iSize, 0);
		vecInvArea.resize (iSize, 1);
		vecResultEqual1.resize (iSize, FALSE);
	}
	s_TemplData ()
	{
		bIsPatternLearned = FALSE;
	}
};
struct s_MatchParameter
{
	Point2d pt;
	double dMatchScore;
	double dMatchAngle;
	//Mat matRotatedSrc;
	Rect rectRoi;
	double dAngleStart;
	double dAngleEnd;
	RotatedRect rectR;
	Rect rectBounding;
	BOOL bDelete;

	double vecResult[3][3];//for subpixel
	int iMaxScoreIndex;//for subpixel
	BOOL bPosOnBorder;
	Point2d ptSubPixel;
	double dNewAngle;

	s_MatchParameter (Point2f ptMinMax, double dScore, double dAngle)//, Mat matRotatedSrc = Mat ())
	{
		pt = ptMinMax;
		dMatchScore = dScore;
		dMatchAngle = dAngle;

		bDelete = FALSE;
		dNewAngle = 0.0;

		bPosOnBorder = FALSE;
	}
	s_MatchParameter ()
	{
		double dMatchScore = 0;
		double dMatchAngle = 0;
	}
	~s_MatchParameter ()
	{

	}
};
struct s_SingleTargetMatch
{
	Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
	double dMatchedAngle;
	double dMatchScore;
};
struct s_BlockMax
{
	struct Block 
	{
		Rect rect;
		double dMax;
		Point ptMaxLoc;
		Block ()
		{}
		Block (Rect rect_, double dMax_, Point ptMaxLoc_)
		{
			rect = rect_;
			dMax = dMax_;
			ptMaxLoc = ptMaxLoc_;
		}
	};
	s_BlockMax ()
	{}
	vector<Block> vecBlock;
	Mat matSrc;
	s_BlockMax (Mat matSrc_, Size sizeTemplate)
	{
		matSrc = matSrc_;
		//將matSrc 拆成數個block，分別計算最大值
		int iBlockW = sizeTemplate.width * 2;
		int iBlockH = sizeTemplate.height * 2;

		int iCol = matSrc.cols / iBlockW;
		BOOL bHResidue = matSrc.cols % iBlockW != 0;

		int iRow = matSrc.rows / iBlockH;
		BOOL bVResidue = matSrc.rows % iBlockH != 0;

		if (iCol == 0 || iRow == 0)
		{
			vecBlock.clear ();
			return;
		}

		vecBlock.resize (iCol * iRow);
		int iCount = 0;
		for (int y = 0; y < iRow ; y++)
		{
			for (int x = 0; x < iCol; x++)
			{
				Rect rectBlock (x * iBlockW, y * iBlockH, iBlockW, iBlockH);
				vecBlock[iCount].rect = rectBlock;
				minMaxLoc (matSrc (rectBlock), 0, &vecBlock[iCount].dMax, 0, &vecBlock[iCount].ptMaxLoc);
				vecBlock[iCount].ptMaxLoc += rectBlock.tl ();
				iCount++;
			}
		}
		if (bHResidue && bVResidue)
		{
			Rect rectRight (iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
			Block blockRight;
			blockRight.rect = rectRight;
			minMaxLoc (matSrc (rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
			blockRight.ptMaxLoc += rectRight.tl ();
			vecBlock.push_back (blockRight);

			Rect rectBottom (0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
			Block blockBottom;
			blockBottom.rect = rectBottom;
			minMaxLoc (matSrc (rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
			blockBottom.ptMaxLoc += rectBottom.tl ();
			vecBlock.push_back (blockBottom);
		}
		else if (bHResidue)
		{
			Rect rectRight (iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
			Block blockRight;
			blockRight.rect = rectRight;
			minMaxLoc (matSrc (rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
			blockRight.ptMaxLoc += rectRight.tl ();
			vecBlock.push_back (blockRight);
		}
		else
		{
			Rect rectBottom (0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
			Block blockBottom;
			blockBottom.rect = rectBottom;
			minMaxLoc (matSrc (rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
			blockBottom.ptMaxLoc += rectBottom.tl ();
			vecBlock.push_back (blockBottom);
		}
	}
	void UpdateMax (Rect rectIgnore)
	{
		if (vecBlock.size () == 0)
			return;
		//找出所有跟rectIgnore交集的block
		int iSize = vecBlock.size ();
		for (int i = 0; i < iSize ; i++)
		{
			Rect rectIntersec = rectIgnore & vecBlock[i].rect;
			//無交集
			if (rectIntersec.width == 0 && rectIntersec.height == 0)
				continue;
			//有交集，更新極值和極值位置
			minMaxLoc (matSrc (vecBlock[i].rect), 0, &vecBlock[i].dMax, 0, &vecBlock[i].ptMaxLoc);
			vecBlock[i].ptMaxLoc += vecBlock[i].rect.tl ();
		}
	}
	void GetMaxValueLoc (double& dMax, Point& ptMaxLoc)
	{
		int iSize = vecBlock.size ();
		if (iSize == 0)
		{
			minMaxLoc (matSrc, 0, &dMax, 0, &ptMaxLoc);
			return;
		}
		//從block中找最大值
		int iIndex = 0;
		dMax = vecBlock[0].dMax;
		for (int i = 1 ; i < iSize; i++)
		{
			if (vecBlock[i].dMax >= dMax)
			{
				iIndex = i;
				dMax = vecBlock[i].dMax;
			}
		}
		ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
	}
};



// CMatchToolDlg 對話方塊
class CMatchToolDlg
{
public:
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MATCHTOOL_DIALOG };
#endif
	int m_iMaxPos;
	double m_dMaxOverlap;
	double m_dScore;
	double m_dToleranceAngle;
	int m_iMinReduceArea;
	cv::Mat m_matSrc;
	cv::Mat m_matDst;
	BOOL m_bDebugMode;
	bool m_bStopLayer1;//FastMode
	BOOL ckSIMD;
	BOOL bSubPixelEstimation;

	BOOL m_bToleranceRange;
	double m_dTolerance1;
	double m_dTolerance2;
	double m_dTolerance3;
	double m_dTolerance4;

	vector<s_SingleTargetMatch> m_vecSingleTargetData;

	CMatchToolDlg();
	BOOL SubPixEsimation (vector<s_MatchParameter>* vec, double* dX, double* dY, double* dAngle, double dAngleStep, int iMaxScoreIndex);
	
	BOOL Match ();
	void LearnPattern ();

	BOOL m_ckBitwiseNot;
	BOOL m_bSubPixel;
	BOOL m_ckSIMD;
	
private:
	s_TemplData m_TemplData;
	int GetTopLayer (Mat* matTempl, int iMinDstLength);

	vector<s_MatchParameter> GetMatchCandidates(vector<Mat> vecMatSrcPyr, int iTopLayer, vector<double> vecAngles, vector<double> vecLayerScore, Point2f ptCenter);

	vector<s_MatchParameter> GetMatches(vector<s_MatchParameter> vecMatchParameter, vector<Mat> vecMatSrcPyr,
	                                    vector<double> vecLayerScore, Point2f ptCenter, int iTopLayer, int iDstW,
	                                    int iDstH);

	void MatchTemplate (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, BOOL bUseSIMD);

	void MatchTemplateInverted(cv::Mat &matSrc, s_TemplData *pTemplData, cv::Mat &matResult, int iLayer,
	                           uint8_t bUseSIMD);

	void GetRotatedROI (Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI);
	void CCOEFF_Denominator (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer);
	void CCOEFF_Denominator_SIMD (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer);

	void CCOEFF_Denominator_SIMD_AVX(cv::Mat &matSrc, s_TemplData *pTemplData, cv::Mat &matResult, int iLayer);

	Size  GetBestRotationSize (Size sizeSrc, Size sizeDst, double dRAngle);
	Point2f ptRotatePt2f (Point2f ptInput, Point2f ptOrg, double dAngle);
	void FilterWithScore (vector<s_MatchParameter>* vec, double dScore);
	void FilterWithRotatedRect (vector<s_MatchParameter>* vec, int iMethod = CV_TM_CCOEFF_NORMED, double dMaxOverLap = 0);
	Point GetNextMaxLoc (Mat & matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap);
	Point GetNextMaxLoc (Mat & matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax);
	void SortPtWithCenter (vector<Point2f>& vecSort);
};

// =============================================================================
// =============================================================================
// =============================================================================

// IMPLEMENTATION

// =============================================================================
// =============================================================================
// =============================================================================

#define VISION_TOLERANCE 0.0000001
#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define MATCH_CANDIDATE_NUM 5

#define SUBITEM_INDEX 0
#define SUBITEM_SCORE 1
#define SUBITEM_ANGLE 2
#define SUBITEM_POS_X 3
#define SUBITEM_POS_Y 4

#define MAX_SCALE_TIMES 10
#define MIN_SCALE_TIMES 0
#define SCALE_RATIO 1.25

#define FONT_SIZE 115
// CMatchToolDlg 對話方塊
bool compareScoreBig2Small (const s_MatchParameter& lhs, const s_MatchParameter& rhs) { return  lhs.dMatchScore > rhs.dMatchScore; }
bool comparePtWithAngle (const pair<Point2f, double> lhs, const pair<Point2f, double> rhs) { return lhs.second < rhs.second; }
bool compareMatchResultByPos (const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs)
{
	double dTol = 2;
	if (fabs (lhs.ptCenter.y - rhs.ptCenter.y) <= dTol)
		return lhs.ptCenter.x < rhs.ptCenter.x;
	else
		return lhs.ptCenter.y < rhs.ptCenter.y;

};
bool compareMatchResultByScore (const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs) { return lhs.dMatchScore > rhs.dMatchScore; }
bool compareMatchResultByPosX (const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs) { return lhs.ptCenter.x < rhs.ptCenter.x; }


void MouseCall (int event, int x, int y, int flag, void* pUserData);
const Scalar colorWaterBlue (230, 255, 102);
const Scalar colorBlue (255, 0, 0);
const Scalar colorYellow (0, 255, 255);
const Scalar colorRed (0, 0, 255);
const Scalar colorBlack (0, 0, 0);
const Scalar colorGray (200, 200, 200);
const Scalar colorSystem (240, 240, 240);
const Scalar colorGreen (0, 255, 0);
const Scalar colorWhite (255, 255, 255);
const Scalar colorPurple (214, 112, 218);
const Scalar colorGoldenrod (15, 185, 255);


//ctor
CMatchToolDlg::CMatchToolDlg()
	: 
	  m_iMaxPos (70)
	, m_dMaxOverlap (0)
	, m_dScore (0.8)
	, m_dToleranceAngle (0)
	, m_iMinReduceArea (256)
	, m_bDebugMode (FALSE)
	, m_dTolerance1 (40)
	, m_dTolerance2 (60)
	, m_dTolerance3 (-110)
	, m_dTolerance4 (-100)
	, m_bStopLayer1(false)
	, m_bToleranceRange(FALSE)
    ,m_ckBitwiseNot(FALSE)
{
}

void CMatchToolDlg::LearnPattern ()
{
	m_TemplData.clear ();

	int iTopLayer = GetTopLayer (&m_matDst, (int)sqrt ((double)m_iMinReduceArea));
	buildPyramid (m_matDst, m_TemplData.vecPyramid, iTopLayer);
	s_TemplData* templData = &m_TemplData;
	templData->iBorderColor = mean (m_matDst).val[0] < 128 ? 255 : 0;
	int iSize = templData->vecPyramid.size ();
	templData->resize (iSize);

	for (int i = 0; i < iSize; i++)
	{
		double invArea = 1. / ((double)templData->vecPyramid[i].rows * templData->vecPyramid[i].cols);
		Scalar templMean, templSdv;
		double templNorm = 0, templSum2 = 0;

		meanStdDev (templData->vecPyramid[i], templMean, templSdv);
		templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

		if (templNorm < DBL_EPSILON)
		{
			templData->vecResultEqual1[i] = TRUE;
		}
		templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];


		templSum2 /= invArea;
		templNorm = std::sqrt (templNorm);
		templNorm /= std::sqrt (invArea); // care of accuracy here


		templData->vecInvArea[i] = invArea;
		templData->vecTemplMean[i] = templMean;
		templData->vecTemplNorm[i] = templNorm;
	}
	templData->bIsPatternLearned = TRUE;
}

int CMatchToolDlg::GetTopLayer (Mat* matTempl, int iMinDstLength)
{

	int iTopLayer = 0;
	int iMinReduceArea = iMinDstLength * iMinDstLength;
	int iArea = matTempl->cols * matTempl->rows;
	while (iArea > iMinReduceArea)
	{
		iArea /= 4;
		iTopLayer++;
	}
	return iTopLayer;
}


vector<s_MatchParameter> CMatchToolDlg::GetMatchCandidates(vector<Mat>  vecMatSrcPyr,int iTopLayer, vector<double> vecAngles, vector<double> vecLayerScore, Point2f ptCenter) {

	s_TemplData* pTemplData = &m_TemplData;
	Size sizePat = pTemplData->vecPyramid[iTopLayer].size ();
	BOOL bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size ().area () / sizePat.area () > 500) && m_iMaxPos > 10;

	vector<s_MatchParameter> vecMatchParameter;
	// #pragma omp parallel for
	for (double vecAngle : vecAngles)
	{
		Mat matRotatedSrc, matR = getRotationMatrix2D (ptCenter, vecAngle, 1);
		Mat matResult;
		Point ptMaxLoc;
		double dValue, dMaxVal;
		double dRotate = clock ();
		Size sizeBest = GetBestRotationSize (vecMatSrcPyr[iTopLayer].size (), pTemplData->vecPyramid[iTopLayer].size (), vecAngle);

		float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
		float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
		matR.at<double> (0, 2) += fTranslationX;
		matR.at<double> (1, 2) += fTranslationY;
		warpAffine (vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, Scalar (pTemplData->iBorderColor));

		MatchTemplate (matRotatedSrc, pTemplData, matResult, iTopLayer, FALSE);

		if (bCalMaxByBlock)
		{
			s_BlockMax blockMax (matResult, pTemplData->vecPyramid[iTopLayer].size ());
			blockMax.GetMaxValueLoc (dMaxVal, ptMaxLoc);
			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;

			#pragma omp critical
			{
				vecMatchParameter.push_back (s_MatchParameter (Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngle));
			}

			for (int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc (matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size (), dValue, m_dMaxOverlap, blockMax);
				if (dValue < vecLayerScore[iTopLayer])
					break;

				#pragma omp critical
				{
					vecMatchParameter.push_back (s_MatchParameter (Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngle));
				}
			}
		}
		else
		{
			minMaxLoc (matResult, 0, &dMaxVal, 0, &ptMaxLoc);
			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;

			#pragma omp critical
			{
				vecMatchParameter.push_back(s_MatchParameter(
					Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngle));
			}

			for (int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc (matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size (), dValue, m_dMaxOverlap);
				if (dValue < vecLayerScore[iTopLayer])
					break;

				#pragma omp critical
				{
					vecMatchParameter.push_back (s_MatchParameter (Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngle));
				}
			}
		}
	}
	sort (vecMatchParameter.begin (), vecMatchParameter.end (), compareScoreBig2Small);

	return vecMatchParameter;
}


vector<s_MatchParameter> CMatchToolDlg::GetMatches(vector<s_MatchParameter> vecMatchParameter, vector<Mat> vecMatSrcPyr,
                                                   vector<double> vecLayerScore, Point2f ptCenter, int iTopLayer,
                                                   int iDstW, int iDstH) {

	//第一階段結束
	BOOL bSubPixelEstimation = m_bSubPixel;
	int iStopLayer = m_bStopLayer1 ? 1 : 0; //设置为1时：粗匹配，牺牲精度提升速度。
	//int iSearchSize = min (m_iMaxPos + MATCH_CANDIDATE_NUM, (int)vecMatchParameter.size ());//可能不需要搜尋到全部 太浪費時間
	vector<s_MatchParameter> vecAllResult;

	s_TemplData* pTemplData = &m_TemplData;


	// #pragma omp parallel for
	for (auto & matchParam : vecMatchParameter)
	//for (int i = 0; i < iSearchSize; i++)
	{
		double dRAngle = -matchParam.dMatchAngle * D2R;
		Point2f ptLT = ptRotatePt2f (matchParam.pt, ptCenter, dRAngle);

		double dAngleStep = atan (2.0 / max (iDstW, iDstH)) * R2D;//min改為max
		matchParam.dAngleStart = matchParam.dMatchAngle - dAngleStep;
		matchParam.dAngleEnd = matchParam.dMatchAngle + dAngleStep;

		if (iTopLayer <= iStopLayer)
		{
			matchParam.pt = Point2d (ptLT * ((iTopLayer == 0) ? 1 : 2));
			#pragma omp critical
			{
				vecAllResult.push_back (matchParam);
			}
		}
		else
		{
			for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
			{
				//搜尋角度
				dAngleStep = atan (2.0 / max (pTemplData->vecPyramid[iLayer].cols, pTemplData->vecPyramid[iLayer].rows)) * R2D;//min改為max
				vector<double> vecAngles;
				//double dAngleS = vecMatchParameter[i].dAngleStart, dAngleE = vecMatchParameter[i].dAngleEnd;
				double dMatchedAngle = matchParam.dMatchAngle;
				if (m_bToleranceRange)
				{
					for (int i = -1; i <= 1; i++)
						vecAngles.push_back (dMatchedAngle + dAngleStep * i);
				}
				else
				{
					if (m_dToleranceAngle < VISION_TOLERANCE)
						vecAngles.push_back (0.0);
					else
						for (int i = -1; i <= 1; i++)
							vecAngles.push_back (dMatchedAngle + dAngleStep * i);
				}
				Point2f ptSrcCenter ((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
				int iSize = (int)vecAngles.size ();
				vector<s_MatchParameter> vecNewMatchParameter (iSize);
				int iMaxScoreIndex = 0;
				double dBigValue = -1;
				for (int j = 0; j < iSize; j++)
				{

					Mat matResult, matRotatedSrc;
					double dMaxValue = 0;
					Point ptMaxLoc;
					GetRotatedROI (vecMatSrcPyr[iLayer], pTemplData->vecPyramid[iLayer].size (), ptLT * 2, vecAngles[j], matRotatedSrc);
					MatchTemplate (matRotatedSrc, pTemplData, matResult, iLayer, TRUE);
					//matchTemplate (matRotatedSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCOEFF_NORMED);
					minMaxLoc (matResult, 0, &dMaxValue, 0, &ptMaxLoc);
					vecNewMatchParameter[j] = s_MatchParameter (ptMaxLoc, dMaxValue, vecAngles[j]);

					if (vecNewMatchParameter[j].dMatchScore > dBigValue)
					{
						iMaxScoreIndex = j;
						dBigValue = vecNewMatchParameter[j].dMatchScore;
					}
					//次像素估計
					if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
						vecNewMatchParameter[j].bPosOnBorder = TRUE;
					if (!vecNewMatchParameter[j].bPosOnBorder)
					{
						for (int y = -1; y <= 1; y++)
							for (int x = -1; x <= 1; x++)
								vecNewMatchParameter[j].vecResult[x + 1][y + 1] = matResult.at<float> (ptMaxLoc + Point (x, y));
					}
					//次像素估計

				}
				if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
					break;
				//次像素估計
				if (bSubPixelEstimation
					&& iLayer == 0
					&& (!vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder)
					&& iMaxScoreIndex != 0
					&& iMaxScoreIndex != 2)
				{
					double dNewX = 0, dNewY = 0, dNewAngle = 0;
					SubPixEsimation ( &vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, dAngleStep, iMaxScoreIndex);
					vecNewMatchParameter[iMaxScoreIndex].pt = Point2d (dNewX, dNewY);
					vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
				}
				//次像素估計

				double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

				//讓坐標系回到旋轉時(GetRotatedROI)的(0, 0)
				Point2f ptPaddingLT = ptRotatePt2f (ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f (3, 3);
				Point2f pt (vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
				//再旋轉
				pt = ptRotatePt2f (pt, ptSrcCenter, -dNewMatchAngle * D2R);

				if (iLayer == iStopLayer)
				{
					vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
					#pragma omp critical
					{
						vecAllResult.push_back (vecNewMatchParameter[iMaxScoreIndex]);
					}
				}
				else
				{
					//更新MatchAngle ptLT
					matchParam.dMatchAngle = dNewMatchAngle;
					matchParam.dAngleStart = matchParam.dMatchAngle - dAngleStep / 2;
					matchParam.dAngleEnd = matchParam.dMatchAngle + dAngleStep / 2;
					ptLT = pt;
				}
			}

		}
	}
	FilterWithScore (&vecAllResult, m_dScore);

	return vecAllResult;
}

//OCR
bool comparePosWithY (const pair<Point2d, char>& lhs, const pair<Point2d, char>& rhs) { return lhs.first.y < rhs.first.y; }
bool comparePosWithX (const pair<Point2d, char>& lhs, const pair<Point2d, char>& rhs){return lhs.first.x < rhs.first.x;}
//OCR

BOOL CMatchToolDlg::Match ()
{
	if (m_matSrc.empty () || m_matDst.empty ())
		return FALSE;
	if ((m_matDst.cols < m_matSrc.cols && m_matDst.rows > m_matSrc.rows) || (m_matDst.cols > m_matSrc.cols && m_matDst.rows < m_matSrc.rows))
		return FALSE;
	if (m_matDst.size ().area () > m_matSrc.size ().area ())
		return FALSE;
	if (!m_TemplData.bIsPatternLearned)
		return FALSE;
	double d1 = clock ();
	//決定金字塔層數 總共為1 + iLayer層
	int iTopLayer = GetTopLayer (&m_matDst, (int)sqrt ((double)m_iMinReduceArea));
	//建立金字塔
	vector<Mat> vecMatSrcPyr;
	if (m_ckBitwiseNot)
	{
		Mat matNewSrc = 255 - m_matSrc;
		buildPyramid (matNewSrc, vecMatSrcPyr, iTopLayer);
		imshow ("1", matNewSrc);
		moveWindow ("1", 0, 0);
	}
	else
		buildPyramid (m_matSrc, vecMatSrcPyr, iTopLayer);

	s_TemplData* pTemplData = &m_TemplData;

	//第一階段以最頂層找出大致角度與ROI
	double dAngleStep = atan (2.0 / max (pTemplData->vecPyramid[iTopLayer].cols, pTemplData->vecPyramid[iTopLayer].rows)) * R2D;

	vector<double> vecAngles;

	if (m_bToleranceRange)
	{
		if (m_dTolerance1 >= m_dTolerance2 || m_dTolerance3 >= m_dTolerance4)
		{
			return FALSE;
		}
		for (double dAngle = m_dTolerance1; dAngle < m_dTolerance2 + dAngleStep; dAngle += dAngleStep)
			vecAngles.push_back (dAngle);
		for (double dAngle = m_dTolerance3; dAngle < m_dTolerance4 + dAngleStep; dAngle += dAngleStep)
			vecAngles.push_back (dAngle);
	}
	else
	{
		if (m_dToleranceAngle < VISION_TOLERANCE)
			vecAngles.push_back (0.0);
		else
		{
			for (double dAngle = 0; dAngle < m_dToleranceAngle + dAngleStep; dAngle += dAngleStep)
				vecAngles.push_back (dAngle);
			for (double dAngle = -dAngleStep; dAngle > -m_dToleranceAngle - dAngleStep; dAngle -= dAngleStep)
				vecAngles.push_back (dAngle);
		}
	}

	int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
	int iDstW = pTemplData->vecPyramid[iTopLayer].cols, iDstH = pTemplData->vecPyramid[iTopLayer].rows;
	Point2f ptCenter ((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);
	vector<double> vecLayerScore (iTopLayer + 1, m_dScore);
	for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
		vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

	printf("PART 1\n");
	auto vecMatchParameter = GetMatchCandidates(vecMatSrcPyr, iTopLayer, vecAngles, vecLayerScore, ptCenter);
	printf("PART 2\n");
	vector<s_MatchParameter> vecAllResult = GetMatches(vecMatchParameter, vecMatSrcPyr, vecLayerScore, ptCenter, iTopLayer, iDstW, iDstH);

	//最後濾掉重疊
	int iStopLayer = m_bStopLayer1 ? 1 : 0; //设置为1时：粗匹配，牺牲精度提升速度。
	iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
	iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

	for (int i = 0; i < (int)vecAllResult.size (); i++)
	{
		Point2f ptLT, ptRT, ptRB, ptLB;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
		ptLT = vecAllResult[i].pt;
		ptRT = Point2f (ptLT.x + iDstW * (float)cos (dRAngle), ptLT.y - iDstW * (float)sin (dRAngle));
		ptLB = Point2f (ptLT.x + iDstH * (float)sin (dRAngle), ptLT.y + iDstH * (float)cos (dRAngle));
		ptRB = Point2f (ptRT.x + iDstH * (float)sin (dRAngle), ptRT.y + iDstH * (float)cos (dRAngle));
		//紀錄旋轉矩形
		vecAllResult[i].rectR = RotatedRect(ptLT, ptRT, ptRB);
	}
	FilterWithRotatedRect (&vecAllResult, CV_TM_CCOEFF_NORMED, m_dMaxOverlap);
	//最後濾掉重疊

	//根據分數排序
	sort (vecAllResult.begin (), vecAllResult.end (), compareScoreBig2Small);
	
	m_vecSingleTargetData.clear ();
	int iMatchSize = (int)vecAllResult.size ();
	if (vecAllResult.size () == 0)
		return FALSE;
	int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;

	for (int i = 0; i < iMatchSize; i++)
	{
		s_SingleTargetMatch sstm;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

		sstm.ptLT = vecAllResult[i].pt;

		sstm.ptRT = Point2d (sstm.ptLT.x + iW * cos (dRAngle), sstm.ptLT.y - iW * sin (dRAngle));
		sstm.ptLB = Point2d (sstm.ptLT.x + iH * sin (dRAngle), sstm.ptLT.y + iH * cos (dRAngle));
		sstm.ptRB = Point2d (sstm.ptRT.x + iH * sin (dRAngle), sstm.ptRT.y + iH * cos (dRAngle));
		sstm.ptCenter = Point2d ((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
		sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
		sstm.dMatchScore = vecAllResult[i].dMatchScore;

		if (sstm.dMatchedAngle < -180)
			sstm.dMatchedAngle += 360;
		if (sstm.dMatchedAngle > 180)
			sstm.dMatchedAngle -= 360;
		m_vecSingleTargetData.push_back (sstm);

		

		//Test Subpixel
		/*Point2d ptLT = vecAllResult[i].ptSubPixel;
		Point2d ptRT = Point2d (sstm.ptLT.x + iW * cos (dRAngle), sstm.ptLT.y - iW * sin (dRAngle));
		Point2d ptLB = Point2d (sstm.ptLT.x + iH * sin (dRAngle), sstm.ptLT.y + iH * cos (dRAngle));
		Point2d ptRB = Point2d (sstm.ptRT.x + iH * sin (dRAngle), sstm.ptRT.y + iH * cos (dRAngle));
		Point2d ptCenter = Point2d ((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
		CString strDiff;strDiff.Format (L"Diff(x, y):%.3f, %.3f", ptCenter.x - sstm.ptCenter.x, ptCenter.y - sstm.ptCenter.y);
		AfxMessageBox (strDiff);*/
		//Test Subpixel
		//存出MATCH ROI
		if (i + 1 == m_iMaxPos)
			break;
	}
	//sort (m_vecSingleTargetData.begin (), m_vecSingleTargetData.end (), compareMatchResultByPosX);
	
	return (int)m_vecSingleTargetData.size ();
}
BOOL CMatchToolDlg::SubPixEsimation (vector<s_MatchParameter>* vec, double* dNewX, double* dNewY, double* dNewAngle, double dAngleStep, int iMaxScoreIndex)
{
	//Az=S, (A.T)Az=(A.T)s, z = ((A.T)A).inv (A.T)s

	Mat matA (27, 10, CV_64F);
	Mat matZ (10, 1, CV_64F);
	Mat matS (27, 1, CV_64F);

	double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
	double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
	double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
	int iRow = 0;
	/*for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int theta = 0; theta <= 2; theta++)
			{*/
	for (int theta = 0; theta <= 2; theta++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				//xx yy tt xy xt yt x y t 1
				//0  1  2  3  4  5  6 7 8 9
				double dX = dX_maxScore + x;
				double dY = dY_maxScore + y;
				//double dT = (*vec)[theta].dMatchAngle + (theta - 1) * dAngleStep;
				double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
				matA.at<double> (iRow, 0) = dX * dX;
				matA.at<double> (iRow, 1) = dY * dY;
				matA.at<double> (iRow, 2) = dT * dT;
				matA.at<double> (iRow, 3) = dX * dY;
				matA.at<double> (iRow, 4) = dX * dT;
				matA.at<double> (iRow, 5) = dY * dT;
				matA.at<double> (iRow, 6) = dX;
				matA.at<double> (iRow, 7) = dY;
				matA.at<double> (iRow, 8) = dT;
				matA.at<double> (iRow, 9) = 1.0;
				matS.at<double> (iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
				iRow++;
#ifdef _DEBUG
				/*string str = format ("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f", dValueA[0], dValueA[1], dValueA[2], dValueA[3], dValueA[4], dValueA[5], dValueA[6], dValueA[7], dValueA[8], dValueA[9]);
				fileA <<  str << endl;
				str = format ("%.6f", dValueS[iRow]);
				fileS << str << endl;*/
#endif
			}
		}
	}
	//求解Z矩陣，得到k0~k9
	//[ x* ] = [ 2k0 k3 k4 ]-1 [ -k6 ]
	//| y* | = | k3 2k1 k5 |   | -k7 |
	//[ t* ] = [ k4 k5 2k2 ]   [ -k8 ]
	
	//solve (matA, matS, matZ, DECOMP_SVD);
	matZ = (matA.t () * matA).inv () * matA.t ()* matS;
	Mat matZ_t;
	transpose (matZ, matZ_t);
	double* dZ = matZ_t.ptr<double> (0);
	Mat matK1 = (Mat_<double> (3, 3) << 
		(2 * dZ[0]), dZ[3], dZ[4], 
		dZ[3], (2 * dZ[1]), dZ[5], 
		dZ[4], dZ[5], (2 * dZ[2]));
	Mat matK2 = (Mat_<double> (3, 1) << -dZ[6], -dZ[7], -dZ[8]);
	Mat matDelta = matK1.inv () * matK2;

	*dNewX = matDelta.at<double> (0, 0);
	*dNewY = matDelta.at<double> (1, 0);
	*dNewAngle = matDelta.at<double> (2, 0) * R2D;
	return TRUE;
}

//From ImageShop
// 4個有符號的32位的數據相加的和。
inline int _mm_hsum_epi32 (__m128i V)      // V3 V2 V1 V0
{
	// 實測這個速度要快些，_mm_extract_epi32最慢。
	__m128i T = _mm_add_epi32 (V, _mm_srli_si128 (V, 8));  // V3+V1   V2+V0  V1  V0  
	T = _mm_add_epi32 (T, _mm_srli_si128 (T, 4));    // V3+V1+V2+V0  V2+V0+V1 V1+V0 V0 
	return _mm_cvtsi128_si32 (T);       // 提取低位 
}

int times =  0;

//From ImageShop
// 4個有符號的32位的數據相加的和。
inline int _mm256_hsum_epi32 (__m256i V)      // V3 V2 V1 V0
{
	// 實測這個速度要快些，_mm_extract_epi32最慢。
	__m256i T = _mm256_hadd_epi32(V, V);
	T = _mm256_hadd_epi32(T, T);
	__m256i P = _mm256_permute2f128_si256 (T, T, 1);
	T = _mm256_add_epi32(P, T);
	return _mm256_cvtsi256_si32 (T);
}

// 基於SSE的字節數據的乘法。
// <param name="Kernel">需要卷積的核矩陣。 </param>
// <param name="Conv">卷積矩陣。 </param>
// <param name="Length">矩陣所有元素的長度。 </param>
inline int IM_Conv_SIMD (unsigned char* pCharKernel, unsigned char *pCharConv, int iLength)
{
	const int iBlockSize = 16, Block = iLength / iBlockSize;
	__m128i SumV = _mm_setzero_si128 ();
	__m128i Zero = _mm_setzero_si128 ();
	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		__m128i SrcK = _mm_loadu_si128 ((__m128i*)(pCharKernel + Y));
		__m128i SrcC = _mm_loadu_si128 ((__m128i*)(pCharConv + Y));
		__m128i SrcK_L = _mm_unpacklo_epi8 (SrcK, Zero);
		__m128i SrcK_H = _mm_unpackhi_epi8 (SrcK, Zero);
		__m128i SrcC_L = _mm_unpacklo_epi8 (SrcC, Zero);
		__m128i SrcC_H = _mm_unpackhi_epi8 (SrcC, Zero);
		__m128i SumT = _mm_add_epi32 (_mm_madd_epi16 (SrcK_L, SrcC_L), _mm_madd_epi16 (SrcK_H, SrcC_H));
		SumV = _mm_add_epi32 (SumV, SumT);
	}
	int Sum = _mm_hsum_epi32 (SumV);
	for (int Y = Block * iBlockSize; Y < iLength; Y++)
	{
		Sum += pCharKernel[Y] * pCharConv[Y];
	}
	return Sum;
}


inline int IM_Conv (unsigned char* pCharKernel, unsigned char *pCharConv, int iLength)
{
	int Sum = 0;
	for (int Y =0; Y < iLength; Y++)
	{
		Sum += pCharKernel[Y] * pCharConv[Y];
	}

	if (times < 100) {
		times++;
	}

	return Sum;
}

void printBytes(__m256i* vec) {
	for (int Y = 0; Y < 32; Y++) {
		printf("%02hhx ", ((char *)vec)[Y]);
	}
	printf("\n");
}

static const char iotaArr64[64] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };
static const __m256i iota64 = *(__m256i*)&iotaArr64;
static const int iotaArr8[8] = {0,1,2,3,4,5,6,7};
static const __m256i iota8 = *(__m256i*)&iotaArr8;
// 基於SSE的字節數據的乘法。
// <param name="Kernel">需要卷積的核矩陣。 </param>
// <param name="Conv">卷積矩陣。 </param>
// <param name="Length">矩陣所有元素的長度。 </param>
inline int IM_Conv_SIMD_AVX (unsigned char* pCharKernel, unsigned char *pCharConv, int iLength)
{
	const int iBlockSize = 32, Block = iLength / iBlockSize;
	__m256i SumV = _mm256_setzero_si256 ();
	__m256i Zero = _mm256_setzero_si256 ();



	for (int Y = 0; Y < iLength; Y += iBlockSize)
	{
		char diff = min(iLength - Y, 127);
		__m256i clownMask = _mm256_set1_epi8(diff);
		// printf("1 %hhx\n", diff);
		// printBytes(&clownMask);
		clownMask = _mm256_sub_epi8(clownMask, iota64);
		// printf("2\n");
		// printBytes(&clownMask);

		clownMask = _mm256_cmpgt_epi8( clownMask, Zero);
		// printf("3\n");
		// printBytes(&clownMask);

		__m256i SrcK = _mm256_loadu_si256((__m256i*)(pCharKernel + Y));
		__m256i SrcC = _mm256_loadu_si256 ((__m256i*)(pCharConv + Y));
		// printBytes(&SrcK);
		// printBytes(&SrcC);

		SrcK = _mm256_and_si256 (SrcK, clownMask);
		SrcC = _mm256_and_si256 (SrcC, clownMask);

		__m256i SrcK_L = _mm256_unpacklo_epi8 (SrcK, Zero);
		__m256i SrcK_H = _mm256_unpackhi_epi8 (SrcK, Zero);
		__m256i SrcC_L = _mm256_unpacklo_epi8 (SrcC, Zero);
		__m256i SrcC_H = _mm256_unpackhi_epi8 (SrcC, Zero);
		__m256i SumT = _mm256_add_epi32 (_mm256_madd_epi16 (SrcK_L, SrcC_L), _mm256_madd_epi16 (SrcK_H, SrcC_H));
		SumV = _mm256_add_epi32 (SumV, SumT);
	}

	int Sum = _mm256_hsum_epi32 (SumV);
	return Sum;
}

template<typename  T>
void printMat(cv::Mat mat) {
	for (int Y = 0; Y < mat.rows; Y++) {
		for (int X = 0; X < mat.cols; X++) {
			std::cout << mat.at<T>(Y, X) << " ";
		}
		printf("\n");
	}
}


//#define ORG

void CMatchToolDlg::MatchTemplate (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, BOOL bUseSIMD)
{
	if (true) {
		MatchTemplateInverted(matSrc, pTemplData, matResult, iLayer, bUseSIMD);
		return;
	}

	if (m_ckSIMD && bUseSIMD)
	{
		//From ImageShop
		matResult.create (matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
			matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
		matResult.setTo (0);
		cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];
		// printf("SRC: %d %d %p TEMPLATE: %d %d %p \n", matSrc.rows, matSrc.cols, matSrc.data, matTemplate.rows, matTemplate.cols, matTemplate.data);

		int  t_r_end = matTemplate.rows, t_r = 0;
		for (int r = 0; r < matResult.rows; r++)
		{
			float* r_matResult = matResult.ptr<float> (r);
			uchar* r_source = matSrc.ptr<uchar> (r);
			uchar* r_template, *r_sub_source;
			for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
			{
				r_template = matTemplate.ptr<uchar> ();
				r_sub_source = r_source;
				for (t_r = 0; t_r < t_r_end; ++t_r)
				{
					r_sub_source += matSrc.cols;
					r_template += matTemplate.cols;
					*r_matResult += IM_Conv(r_template, r_sub_source, matTemplate.cols);
				}
			}
		}

		//From ImageShop
	}
	else
		matchTemplate (matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);
	
	/*Mat diff;
	absdiff(matResult, matResult, diff);
	double dMaxValue;
	minMaxLoc(diff, 0, &dMaxValue, 0,0);*/
	CCOEFF_Denominator_SIMD_AVX (matSrc, pTemplData, matResult, iLayer);
	// CCOEFF_Denominator (matSrc, pTemplData, matResult, iLayer);
}


void CMatchToolDlg::MatchTemplateInverted (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, BOOL bUseSIMD)
{
	if (m_ckSIMD && bUseSIMD)
	{
		// printf("SIMD\n");
		//From ImageShop
		cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];
		matResult.create (matSrc.rows - matTemplate.rows + 1,matSrc.cols - matTemplate.cols + 1, CV_32FC1);
		matResult.setTo (0);

		// static const int iotaArr[8] = {0,1,2,3,4,5,6,7};


		// printf("SRC: %d %d %p TEMPLATE: %d %d %p \n", matSrc.rows, matSrc.cols, matSrc.data, matTemplate.rows, matTemplate.cols, matTemplate.data);
		if (false)
			for (int rr = 0; rr < matResult.rows; rr++) {
				auto* r_res = matResult.ptr<float> (rr);
				for (int rc = 0; rc < matResult.cols; rc++) {
					for (int tr = 0; tr < matTemplate.rows; tr++) {

						auto* r_src = matSrc.ptr<uchar>(rr + tr);
						auto* r_templ = matTemplate.ptr<uchar>(tr);

						int blockSize = 32;
						for (int tc = 0; tc < matTemplate.cols; tc+=blockSize) {
							__m256i templ = _mm256_cvtepu8_epi16(_mm_loadu_epi8(r_templ + tc));
							__m256i src = _mm256_cvtepu8_epi16(_mm_loadu_epi8(r_src + rc + tc ));
							__m256i mult = _mm256_mullo_epi16(templ, src);

							// __m256i vec =


							r_res[rc] += r_templ[tc] * r_src[rc + tc];
						}
					}

				}
			}


		if (true)
			for(int tr = 0; tr < matTemplate.rows; tr++) {
				uchar* r_templ = matTemplate.ptr<uchar> (tr);
				for (int rr = 0; rr < matResult.cols; rr++) {
					float* r_res = matResult.ptr<float> (rr);
					uchar* r_src = matSrc.ptr<uchar> (tr + rr);
					for (int tc = 0; tc < matTemplate.cols; tc++) {
						int blockSize = 8;
						int blocks = matResult.cols / blockSize;

						__m256i vec_templ = _mm256_set1_epi32(r_templ[tc]);
						for (int rc = 0; rc < matResult.cols; rc+=blockSize) {
							__m256i vec_src = _mm256_cvtepu8_epi32(_mm_loadu_si64(r_src + tc + rc ));
							__m256 vec_mul = _mm256_cvtepi32_ps(_mm256_mul_epi32(vec_src, vec_templ));
							__m256 vec_res = _mm256_loadu_ps(r_res + rc);
							__m256 vec_sum = _mm256_add_ps(vec_mul, vec_res);

							__m256i mask = _mm256_set1_epi32 (rc);
							mask = _mm256_add_epi32 (mask, iota8);
							mask = _mm256_cmpgt_epi32 (_mm256_set1_epi32(matResult.cols), mask);

							_mm256_maskstore_ps(r_res + rc, mask, vec_sum);
							_mm256_storeu_ps(r_res + rc, vec_sum);
							// r_res[rc] += r_templ[tc] * r_src[rc + tc];
							// r_res[rc+1] += r_templ[tc] * r_src[rc + tc+1];
							// r_res[rc+2] += r_templ[tc] * r_src[rc + tc+2];
							// r_res[rc+3] += r_templ[tc] * r_src[rc + tc+3];
							// r_res[rc+4] += r_templ[tc] * r_src[rc + tc+4];
							// r_res[rc+5] += r_templ[tc] * r_src[rc + tc+5];
							// r_res[rc+6] += r_templ[tc] * r_src[rc + tc+6];
							// r_res[rc+7] += r_templ[tc] * r_src[rc + tc+7];
						}
					}
				}
			}

		if (false)
			for (int sr = 0; sr < matSrc.rows; sr++) {
				uchar* r_src = matSrc.ptr<uchar> (sr);
				for (int sc = 0; sc < matSrc.cols; sc++) {
					for (int tr = max(0, sr - matResult.rows); tr < sr + 1 && tr < matTemplate.rows; tr++) {
						uchar* r_templ = matTemplate.ptr<uchar> (tr);
						int rr = sr - tr;
						float* r_res = matResult.ptr<float> (rr);
						for (int tc = max(0, sc - matResult.cols); tc < min(sc + 1, matTemplate.cols); tc++) {
							int rc = sc - tc;
							r_res[rc] += r_templ[tc] * r_src[sc];
						}
					}
				}
			}

		// printf("\n");
		// printMat<float>(matResult);
	}
	else
		matchTemplate (matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);

	/*Mat diff;
	absdiff(matResult, matResult, diff);
	double dMaxValue;
	minMaxLoc(diff, 0, &dMaxValue, 0,0);*/
	CCOEFF_Denominator_SIMD_AVX (matSrc, pTemplData, matResult, iLayer);
	// CCOEFF_Denominator (matSrc, pTemplData, matResult, iLayer);
}




void CMatchToolDlg::GetRotatedROI (Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI)
{
	double dAngle_radian = dAngle * D2R;
	Point2f ptC ((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
	Point2f ptLT_rotate = ptRotatePt2f (ptLT, ptC, dAngle_radian);
	Size sizePadding (size.width + 6, size.height + 6);


	Mat rMat = getRotationMatrix2D (ptC, dAngle, 1);
	rMat.at<double> (0, 2) -= ptLT_rotate.x - 3;
	rMat.at<double> (1, 2) -= ptLT_rotate.y - 3;
	//平移旋轉矩陣(0, 2) (1, 2)的減，為旋轉後的圖形偏移，-= ptLT_rotate.x - 3 代表旋轉後的圖形往-X方向移動ptLT_rotate.x - 3
	//Debug
	
	//Debug
	warpAffine (matSrc, matROI, rMat, sizePadding);
}


void CMatchToolDlg::CCOEFF_Denominator (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
{
	// auto t2 = Timer("t2");
	if (pTemplData->vecResultEqual1[iLayer])
	{
		matResult = Scalar::all (1);
		return;
	}

	Mat sum, sqsum;
	integral (matSrc, sum, sqsum, CV_64F);

	double* q0 = (double*)sqsum.data;
	double* q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
	double* q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
	double* q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

	double* p0 = (double*)sum.data;
	double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
	double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows*sum.step);
	double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

	int sumstep = sum.data ? (int)(sum.step / sizeof (double)) : 0;
	int sqstep = sqsum.data ? (int)(sqsum.step / sizeof (double)) : 0;

	//
	double dTemplMean = pTemplData->vecTemplMean[iLayer][0];
	double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
	double dInvArea = pTemplData->vecInvArea[iLayer];

	{
		// auto t1 = Timer("t1");
		for (int i = 0; i < matResult.rows; i++)
		{
			float* rrow = matResult.ptr<float> (i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for (int j = 0; j < matResult.cols; j += 1)
			{

				double t = p0[idx + j] - p1[idx + j] - p2[idx + j] + p3[idx + j];
				double num = (double)rrow[j] - t * dTemplMean;
				double wndMean = t * t * dInvArea;
				double wndSum2 = q0[idx2 + j] - q1[idx + j] - q2[idx2 + j] + q3[idx2 + j];
				double threshold = 0;
				double diff = MAX (wndSum2 - wndMean, 0);

				if (diff <= std::min (0.5, 10 * FLT_EPSILON * wndSum2))
					threshold = 0; // avoid rounding errors
				else
					threshold = std::sqrt (diff)*dTemplNorm;

				if (fabs (num) < threshold)
					num /= threshold;
				else if (fabs (num) < threshold * 1.125)
					num = num > 0 ? 1 : -1;
				else
					num = 0;
				// printf("res d: %lf\n", num);

				rrow[j] = (float)num;
				// printres(rrow[j]);
			}
		}
	}
}

static float blend(float* a, float* b, unsigned int mask) {
	return (*reinterpret_cast<unsigned int*>(a) & (~mask)) | (*reinterpret_cast<unsigned int*>(b) & ~mask);
}

void print128_num(__m128i var)
{
	uint32_t *val = (uint32_t*) &var;//can also use uint32_t instead of 16_t
	printf("Numerical: %i %i %i %i \n",val[0], val[1], val[2], val[3]);
}

inline void CMatchToolDlg::CCOEFF_Denominator_SIMD_AVX (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
{
	// auto t2 = Timer("t2");

	if (pTemplData->vecResultEqual1[iLayer])
	{
		matResult = Scalar::all (1);
		return;
	}


	Mat sum, sqsum;
	integral (matSrc, sum, sqsum, CV_64F);

	double* q0 = (double*)sqsum.data;
	double* q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
	double* q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
	double* q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

	double* p0 = (double*)sum.data;
	double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
	double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows*sum.step);
	double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

	int sumstep = sum.data ? (int)(sum.step / sizeof (double)) : 0;
	int sqstep = sqsum.data ? (int)(sqsum.step / sizeof (double)) : 0;

	//
	double dTemplMean = pTemplData->vecTemplMean[iLayer][0];
	double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
	double dInvArea = pTemplData->vecInvArea[iLayer];

	const __m256d zeros = _mm256_setzero_pd();
	const __m128i zerosi = (__m128i)_mm_setzero_pd();
	const __m256d ones = _mm256_set1_pd(1.0f);
	const __m256d minus_ones = _mm256_set1_pd(-1.0f);
	const __m256d vec_dInvArea = _mm256_set1_pd(dInvArea);
	const __m256d vec_dTemplMean = _mm256_set1_pd(dTemplMean);
	const __m256d epsilon = _mm256_set1_pd(10 * FLT_EPSILON);
	const __m256d halves = _mm256_set1_pd(0.5);
	static const int iotaArr[4] = {0,1,2,3};
	const __m128i iota = *(__m128i*)&iotaArr;
	{
		// auto t2 = Timer("t1");
		for (int i = 0; i < matResult.rows; i++)
		{
			auto* rrow = matResult.ptr<float> (i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			int blockSize = 4;
			int blocks = matResult.cols / blockSize;
			int hasExtra =  (matResult.cols - blockSize * blocks) > 0 ? 1 : 0;

			for (int j=0; j < matResult.cols; j+=blockSize)
			{
				// double t = p0[idx + j] - p1[idx + j] - p2[idx + j] + p3[idx + j];
				__m256d vec_p0 = _mm256_loadu_pd(&p0[idx +j]);
				__m256d vec_p1 = _mm256_loadu_pd(&p1[idx +j]);
				__m256d vec_p2 = _mm256_loadu_pd(&p2[idx +j]);
				__m256d vec_p3 = _mm256_loadu_pd(&p3[idx +j]);
				__m256d t = _mm256_add_pd(vec_p0, vec_p3);
				t = _mm256_sub_pd(t, vec_p1);
				t = _mm256_sub_pd(t, vec_p2);

				// printvec("t", t);

				// double num = rrow[j] - t * dTemplMean;
				__m128 numFloats = _mm_loadu_ps(&rrow[j]);
				__m256d vec_num = _mm256_cvtps_pd(numFloats); // Expensive cast to doubles... can we prevent that somehow?
				vec_num = _mm256_fnmadd_pd(t, vec_dTemplMean, vec_num);
				// printvec("num", vec_num);

				// double wndMean = t * t * dInvArea;
				__m256d vec_wndMean = _mm256_mul_pd(_mm256_mul_pd(t, t), vec_dInvArea);
				// printvec("vec_wndMean", vec_wndMean);

				// double wndSum2 = q0[idx2 + j] - q1[idx + j] - q2[idx2 + j] + q3[idx2 + j];
				__m256d vec_q0 = _mm256_loadu_pd(&q0[idx2 +j]);
				__m256d vec_q1 = _mm256_loadu_pd(&q1[idx2 +j]);
				__m256d vec_q2 = _mm256_loadu_pd(&q2[idx2 +j]);
				__m256d vec_q3 = _mm256_loadu_pd(&q3[idx2 +j]);
				__m256d vec_wndSum2 = _mm256_add_pd(vec_q0, vec_q3);
				vec_wndSum2 = _mm256_sub_pd(vec_wndSum2, vec_q1);
				vec_wndSum2 = _mm256_sub_pd(vec_wndSum2, vec_q2);
				// printvec("vec_wndSum2", vec_wndSum2);

				// double diff = MAX (wndSum2 - wndMean, 0);
				__m256d vec_diff = _mm256_max_pd(_mm256_sub_pd(vec_wndSum2, vec_wndMean), zeros);
				// std::min (0.5, 10 * FLT_EPSILON * wndSum2)
				__m256d vec_diffThresh = _mm256_min_pd(halves, _mm256_mul_pd( vec_wndSum2,epsilon ));

				// printvec("vec_diff", vec_diff);
				// printvec("vec_diffThresh", vec_diffThresh);

				// double threshold = 0;
				// if (diff <= std::min (0.5, 10 * FLT_EPSILON * wndSum2))
				// 	threshold = 0; // avoid rounding errors
				// else
				// 	threshold = std::sqrt (diff)*dTemplNorm;

				__m256d threshold_val_false = _mm256_mul_pd(_mm256_sqrt_pd(vec_diff), _mm256_set1_pd(dTemplNorm));
				__m256d vec_threshold_mask = _mm256_cmp_pd(vec_diff, vec_diffThresh, _CMP_LE_OS);
				__m256d vec_threshold = _mm256_andnot_pd(vec_threshold_mask, threshold_val_false);
				// printvec("vec_threshold", vec_threshold);


				// fabs (num)
				// ( from https://stackoverflow.com/questions/5508628/how-to-absolute-2-double-or-4-floats-using-sse-instruction-set-up-to-sse4)
				__m256d sign_mask = _mm256_set1_pd(-0.f);
				__m256d vec_num_abs = _mm256_andnot_pd(sign_mask, vec_num);
				// abs (num) < threshold
				__m256d mask_threshold = _mm256_cmp_pd(vec_num_abs, vec_threshold, _CMP_LT_OS);
				// fabs (num) < threshold * 1.125
				__m256d mask_threshold_mult = _mm256_cmp_pd(vec_num_abs, _mm256_mul_pd(vec_threshold, _mm256_set1_pd(1.125)), _CMP_LT_OS);
				//  num > 0
				__m256d mask_gt_zero = _mm256_cmp_pd(vec_num, zeros, _CMP_GT_OS);
				// num / threshold
				__m256d vec_num_div_thresh = _mm256_div_pd(vec_num, vec_threshold);

				// if (fabs (num) < threshold)
				// 	num /= threshold;
				// else if (fabs (num) < threshold * 1.125)
				// 	num = num > 0 ? 1 : -1;
				// else
				// 	num = 0;
				vec_num = _mm256_or_pd(
					_mm256_and_pd(vec_num_div_thresh,mask_threshold),
				_mm256_or_pd(
					_mm256_and_pd(ones ,_mm256_andnot_pd(mask_threshold, _mm256_and_pd( mask_gt_zero, mask_threshold_mult))),
				_mm256_or_pd(
						_mm256_and_pd(minus_ones ,_mm256_andnot_pd(mask_threshold, _mm256_andnot_pd(mask_gt_zero, mask_threshold_mult))),
						zeros
					)
				));

				// printvec("vec_num", vec_num);
				__m128 result = _mm256_cvtpd_ps(vec_num);

				// This is to prevent overflows when the block size does not divide
				// for example we have cols = 19, blocksize = 4
				// So blocks = 4
				// At the last iterations we have j = 4, and blocksize = 4
				// So we are going to write at adresses j*4, j*4+1 ,j*4+2, j*4+3,
				// which is not good, since addr j*4+3 is out of bounds
				// To fix that we need to mask out any loads that happen outside the bounds
				__m128i clownMask = _mm_set1_epi32(matResult.cols - j); // -1 is important
				clownMask = _mm_sub_epi32(clownMask, iota);
				clownMask = _mm_cmplt_epi32(zerosi, clownMask);
				// print128_num( clownMask);
				// printf("result: %f %f %f %f\n", rrow[j+0], rrow[j+1], rrow[j+2], rrow[j+3]);
				// printf("result: %f %f %f %f\n", result[0], result[1], result[2], result[3]);

				_mm_maskstore_ps(&rrow[j], clownMask, result);
				// _mm_storeu_ps(&rrow[j], result);
				// printf("After: %f %f %f %f\n", rrow[j], rrow[j + 1], rrow[j + 2], rrow[j + 3]);
			}
		}
	}
}
inline void CMatchToolDlg::CCOEFF_Denominator_SIMD (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
{
	if (pTemplData->vecResultEqual1[iLayer])
	{
		matResult = Scalar::all (1);
		return;
	}


	Mat sum, sqsum;
	integral (matSrc, sum, sqsum, CV_64F);

	double* q0 = (double*)sqsum.data;
	double* q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
	double* q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
	double* q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

	double* p0 = (double*)sum.data;
	double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
	double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows*sum.step);
	double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

	int sumstep = sum.data ? (int)(sum.step / sizeof (double)) : 0;
	int sqstep = sqsum.data ? (int)(sqsum.step / sizeof (double)) : 0;

	//
	double dTemplMean = pTemplData->vecTemplMean[iLayer][0];
	double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
	double dInvArea = pTemplData->vecInvArea[iLayer];


	for (int i = 0; i < matResult.rows; i++)
	{
		float* rrow = matResult.ptr<float> (i);
		int idx = i * sumstep;
		int idx2 = i * sqstep;

		int blockSize = 2;
		int blocks = matResult.cols / blockSize;
		int remainder = blocks*blockSize == matResult.cols ? 0 : 1;
		unsigned int add_mask  = blocks*blockSize == matResult.cols ? 0x0 : 0xFFFFFFFF;
		for (int j=0; j < blocks + remainder; j++)
		{

			// double t = p0[idx + j] - p1[idx + j] - p2[idx + j] + p3[idx + j];
			__m128d vec_p0 = _mm_loadu_pd(&p0[idx +j*blockSize]);
			__m128d vec_p1 = _mm_loadu_pd(&p1[idx +j*blockSize]);
			__m128d vec_p2 = _mm_loadu_pd(&p2[idx +j*blockSize]);
			__m128d vec_p3 = _mm_loadu_pd(&p3[idx +j*blockSize]);
			__m128d t = _mm_add_pd(vec_p0, vec_p3);

			t = _mm_sub_pd(t, vec_p1);
			t = _mm_sub_pd(t, vec_p2);

			// printf("t: %lf\n", t[0]);

			// double num = rrow[j] - t * dTemplMean;
			const __m128d zeros = _mm_setzero_pd();

			__m128d vec_num = _mm_cvtps_pd(_mm_loadl_pi( reinterpret_cast<__m128>(zeros), reinterpret_cast<__m64 *>(rrow + j * blockSize)));
			__m128d tmp = _mm_mul_pd(t, _mm_set1_pd(dTemplMean));
			// printf("tmp: %lf %f %f\n", tmp[0], rrow[j*blockSize], vec_num[0]);
			vec_num = _mm_sub_pd(vec_num, tmp);
			// printf("num: %lf\n", vec_num[0]);

			// double wndMean = t * t * dInvArea;
			__m128d vec_wndMean = _mm_mul_pd(_mm_mul_pd(t, t), _mm_set1_pd(dInvArea));
			// printf("vec_wndMean: %lf\n", vec_wndMean[0]);

			// double wndSum2 = q0[idx2 + j] - q1[idx + j] - q2[idx2 + j] + q3[idx2 + j];
			__m128d vec_q0 = _mm_loadu_pd(&q0[idx2 +j*blockSize]);
			__m128d vec_q1 = _mm_loadu_pd(&q1[idx2 +j*blockSize]);
			__m128d vec_q2 = _mm_loadu_pd(&q2[idx2 +j*blockSize]);
			__m128d vec_q3 = _mm_loadu_pd(&q3[idx2 +j*blockSize]);
			__m128d vec_wndSum2 = _mm_add_pd(vec_q0, vec_q3);
			vec_wndSum2 = _mm_sub_pd(vec_wndSum2, vec_q1);
			vec_wndSum2 = _mm_sub_pd(vec_wndSum2, vec_q2);
			// printf("vec_wndSum2: %lf\n", vec_wndSum2[0]);

			// double diff = MAX (wndSum2 - wndMean, 0);
			__m128d vec_diff = _mm_max_pd(_mm_sub_pd(vec_wndSum2, vec_wndMean), zeros);
			// std::min (0.5, 10 * FLT_EPSILON * wndSum2)
			__m128d vec_diffThresh = _mm_min_pd(_mm_set1_pd(0.5), _mm_mul_pd(_mm_set1_pd(10 * FLT_EPSILON), vec_wndSum2 ));

			// printf("vec_diff: %lf\n", vec_diff[0]);
			// printf("vec_diffThresh: %lf\n", vec_diffThresh[0]);

			// double threshold = 0;
			// if (diff <= std::min (0.5, 10 * FLT_EPSILON * wndSum2))
			// 	threshold = 0; // avoid rounding errors
			// else
			// 	threshold = std::sqrt (diff)*dTemplNorm;

			__m128d threshold_val_false = _mm_mul_pd(_mm_sqrt_pd(vec_diff), _mm_set1_pd(dTemplNorm));
			__m128d vec_threshold_mask = _mm_cmple_pd(vec_diff, vec_diffThresh);
			__m128d vec_threshold = _mm_andnot_pd(vec_threshold_mask, threshold_val_false);
			// printf("vec_threshold: %lf\n", vec_threshold[0]);


			// fabs (num)
			// ( from https://stackoverflow.com/questions/5508628/how-to-absolute-2-double-or-4-floats-using-sse-instruction-set-up-to-sse4)
			__m128d sign_mask = _mm_set1_pd(-0.f);
			__m128d vec_num_abs = _mm_andnot_pd(sign_mask, vec_num);
			// abs (num) < threshold
			__m128d mask_theshold = _mm_cmplt_pd(vec_num_abs, vec_threshold);
			// fabs (num) < threshold * 1.125
			__m128d mask_theshold_mult = _mm_cmplt_pd(vec_num_abs, _mm_mul_pd(vec_threshold, _mm_set1_pd(1.125)));
			//  num > 0
			__m128d mask_gt_zero = _mm_cmpgt_pd(vec_num, zeros);
			// num / threshold
			__m128d vec_num_div_thresh = _mm_div_pd(vec_num, vec_threshold);


			// if (fabs (num) < threshold)
			// 	num /= threshold;
			// else if (fabs (num) < threshold * 1.125)
			// 	num = num > 0 ? 1 : -1;
			// else
			// 	num = 0;
			// Not the most efficient but I do not care
			__m128d ones = _mm_set1_pd(1.0f);
			__m128d minus_ones = _mm_set1_pd(-1.0f);
			vec_num = _mm_or_pd(
				_mm_and_pd(vec_num_div_thresh,mask_theshold),
			_mm_or_pd(
				_mm_and_pd(ones ,_mm_andnot_pd(mask_theshold, _mm_and_pd( mask_gt_zero, mask_theshold_mult))),
			_mm_or_pd(
					_mm_and_pd(minus_ones ,_mm_andnot_pd(mask_theshold, _mm_andnot_pd(mask_gt_zero, mask_theshold_mult))),
					zeros
				)
			));
			__m128 result = _mm_cvtpd_ps(vec_num);

			rrow[j*blockSize] = vec_num[0];
			float myFloat = (static_cast<float>(vec_num[1]));
			// This is a buffer overflow :)
			rrow[j*blockSize+1] = blend(&rrow[j*blockSize+1], &myFloat , add_mask);
		}
	}
}



Size CMatchToolDlg::GetBestRotationSize (Size sizeSrc, Size sizeDst, double dRAngle)
{
	double dRAngle_radian = dRAngle * D2R;
	Point ptLT (0, 0), ptLB (0, sizeSrc.height - 1), ptRB (sizeSrc.width - 1, sizeSrc.height - 1), ptRT (sizeSrc.width - 1, 0);
	Point2f ptCenter ((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
	Point2f ptLT_R = ptRotatePt2f (Point2f (ptLT), ptCenter, dRAngle_radian);
	Point2f ptLB_R = ptRotatePt2f (Point2f (ptLB), ptCenter, dRAngle_radian);
	Point2f ptRB_R = ptRotatePt2f (Point2f (ptRB), ptCenter, dRAngle_radian);
	Point2f ptRT_R = ptRotatePt2f (Point2f (ptRT), ptCenter, dRAngle_radian);

	float fTopY = max (max (ptLT_R.y, ptLB_R.y), max (ptRB_R.y, ptRT_R.y));
	float fBottomY = min (min (ptLT_R.y, ptLB_R.y), min (ptRB_R.y, ptRT_R.y));
	float fRightX = max (max (ptLT_R.x, ptLB_R.x), max (ptRB_R.x, ptRT_R.x));
	float fLeftX = min (min (ptLT_R.x, ptLB_R.x), min (ptRB_R.x, ptRT_R.x));

	if (dRAngle > 360)
		dRAngle -= 360;
	else if (dRAngle < 0)
		dRAngle += 360;

	if (fabs (fabs (dRAngle) - 90) < VISION_TOLERANCE || fabs (fabs (dRAngle) - 270) < VISION_TOLERANCE)
	{
		return Size (sizeSrc.height, sizeSrc.width);
	}
	else if (fabs (dRAngle) < VISION_TOLERANCE || fabs (fabs (dRAngle) - 180) < VISION_TOLERANCE)
	{
		return sizeSrc;
	}
	
	double dAngle = dRAngle;

	if (dAngle > 0 && dAngle < 90)
	{
		;
	}
	else if (dAngle > 90 && dAngle < 180)
	{
		dAngle -= 90;
	}
	else if (dAngle > 180 && dAngle < 270)
	{
		dAngle -= 180;
	}
	else if (dAngle > 270 && dAngle < 360)
	{
		dAngle -= 270;
	}


	float fH1 = sizeDst.width * sin (dAngle * D2R) * cos (dAngle * D2R);
	float fH2 = sizeDst.height * sin (dAngle * D2R) * cos (dAngle * D2R);

	int iHalfHeight = (int)ceil (fTopY - ptCenter.y - fH1);
	int iHalfWidth = (int)ceil (fRightX - ptCenter.x - fH2);
	
	Size sizeRet (iHalfWidth * 2, iHalfHeight * 2);

	BOOL bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
		|| (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height
			|| sizeDst.area () > sizeRet.area ());
	if (bWrongSize)
		sizeRet = Size (int (fRightX - fLeftX + 0.5), int (fTopY - fBottomY + 0.5));

	return sizeRet;
}
Point2f CMatchToolDlg::ptRotatePt2f (Point2f ptInput, Point2f ptOrg, double dAngle)
{
	double dWidth = ptOrg.x * 2;
	double dHeight = ptOrg.y * 2;
	double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

	double dX = (ptInput.x - ptOrg.x) * cos (dAngle) - (dY1 - ptOrg.y) * sin (dAngle) + ptOrg.x;
	double dY = (ptInput.x - ptOrg.x) * sin (dAngle) + (dY1 - ptOrg.y) * cos (dAngle) + dY2;

	dY = -dY + dHeight;
	return Point2f ((float)dX, (float)dY);
}
void CMatchToolDlg::FilterWithScore (vector<s_MatchParameter>* vec, double dScore)
{
	sort (vec->begin (), vec->end (), compareScoreBig2Small);
	int iSize = vec->size (), iIndexDelete = iSize + 1;
	for (int i = 0; i < iSize; i++)
	{
		if ((*vec)[i].dMatchScore < dScore)
		{
			iIndexDelete = i;
			break;
		}
	}
	if (iIndexDelete == iSize + 1)//沒有任何元素小於dScore
		return;
	vec->erase (vec->begin () + iIndexDelete, vec->end ());
	return;
}
void CMatchToolDlg::FilterWithRotatedRect (vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap)
{
	int iMatchSize = (int)vec->size ();
	RotatedRect rect1, rect2;
	for (int i = 0; i < iMatchSize - 1; i++)
	{
		if (vec->at (i).bDelete)
			continue;
		for (int j = i + 1; j < iMatchSize; j++)
		{
			if (vec->at (j).bDelete)
				continue;
			rect1 = vec->at (i).rectR;
			rect2 = vec->at (j).rectR;
			vector<Point2f> vecInterSec;
			int iInterSecType = rotatedRectangleIntersection (rect1, rect2, vecInterSec);
			if (iInterSecType == INTERSECT_NONE)//無交集
				continue;
			else if (iInterSecType == INTERSECT_FULL) //一個矩形包覆另一個
			{
				int iDeleteIndex;
				if (iMethod == CV_TM_SQDIFF)
					iDeleteIndex = (vec->at (i).dMatchScore <= vec->at (j).dMatchScore) ? j : i;
				else
					iDeleteIndex = (vec->at (i).dMatchScore >= vec->at (j).dMatchScore) ? j : i;
				vec->at (iDeleteIndex).bDelete = TRUE;
			}
			else//交點 > 0
			{
				if (vecInterSec.size () < 3)//一個或兩個交點
					continue;
				else
				{
					int iDeleteIndex;
					//求面積與交疊比例
					SortPtWithCenter (vecInterSec);
					double dArea = contourArea (vecInterSec);
					double dRatio = dArea / rect1.size.area ();
					//若大於最大交疊比例，選分數高的
					if (dRatio > dMaxOverLap)
					{
						if (iMethod == CV_TM_SQDIFF)
							iDeleteIndex = (vec->at (i).dMatchScore <= vec->at (j).dMatchScore) ? j : i;
						else
							iDeleteIndex = (vec->at (i).dMatchScore >= vec->at (j).dMatchScore) ? j : i;
						vec->at (iDeleteIndex).bDelete = TRUE;
					}
				}
			}
		}
	}
	vector<s_MatchParameter>::iterator it;
	for (it = vec->begin (); it != vec->end ();)
	{
		if ((*it).bDelete)
			it = vec->erase (it);
		else
			++it;
	}
}
Point CMatchToolDlg::GetNextMaxLoc (Mat & matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap)
{
	//比對到的區域完全不重疊 : +-一個樣板寬高
	//int iStartX = ptMaxLoc.x - iTemplateW;
	//int iStartY = ptMaxLoc.y - iTemplateH;
	//int iEndX = ptMaxLoc.x + iTemplateW;

	//int iEndY = ptMaxLoc.y + iTemplateH;
	////塗黑
	//rectangle (matResult, Rect (iStartX, iStartY, 2 * iTemplateW * (1-dMaxOverlap * 2), 2 * iTemplateH * (1-dMaxOverlap * 2)), Scalar (dMinValue), CV_FILLED);
	////得到下一個最大值
	//Point ptNewMaxLoc;
	//minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	//return ptNewMaxLoc;

	//比對到的區域需考慮重疊比例
	int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
	int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);
	//塗黑
	rectangle (matResult, Rect (iStartX, iStartY, 2 * sizeTemplate.width * (1- dMaxOverlap), 2 * sizeTemplate.height * (1- dMaxOverlap)), Scalar (-1), CV_FILLED);
	//得到下一個最大值
	Point ptNewMaxLoc;
	minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	return ptNewMaxLoc;
}
Point CMatchToolDlg::GetNextMaxLoc (Mat & matResult, Point ptMaxLoc, Size sizeTemplate, double & dMaxValue, double dMaxOverlap, s_BlockMax & blockMax)
{
	//比對到的區域需考慮重疊比例
	int iStartX = int (ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
	int iStartY = int (ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
	Rect rectIgnore (iStartX, iStartY, int (2 * sizeTemplate.width * (1 - dMaxOverlap))
		, int (2 * sizeTemplate.height * (1 - dMaxOverlap)));
	//塗黑
	rectangle (matResult, rectIgnore , Scalar (-1), CV_FILLED);
	blockMax.UpdateMax (rectIgnore);
	Point ptReturn;
	blockMax.GetMaxValueLoc (dMaxValue, ptReturn);
	return ptReturn;
}
void CMatchToolDlg::SortPtWithCenter (vector<Point2f>& vecSort)
{
	int iSize = (int)vecSort.size ();
	Point2f ptCenter;
	for (int i = 0; i < iSize; i++)
		ptCenter += vecSort[i];
	ptCenter /= iSize;

	Point2f vecX (1, 0);

	vector<pair<Point2f, double>> vecPtAngle (iSize);
	for (int i = 0; i < iSize; i++)
	{
		vecPtAngle[i].first = vecSort[i];//pt
		Point2f vec1 (vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
		float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
		float fDot = vec1.x;

		if (vec1.y < 0)//若點在中心的上方
		{
			vecPtAngle[i].second = acos (fDot / fNormVec1) * R2D;
		}
		else if (vec1.y > 0)//下方
		{
			vecPtAngle[i].second = 360 - acos (fDot / fNormVec1) * R2D;
		}
		else//點與中心在相同Y
		{
			if (vec1.x - ptCenter.x > 0)
				vecPtAngle[i].second = 0;
			else
				vecPtAngle[i].second = 180;
		}

	}
	sort (vecPtAngle.begin (), vecPtAngle.end (), comparePtWithAngle);
	for (int i = 0; i < iSize; i++)
		vecSort[i] = vecPtAngle[i].first;
}