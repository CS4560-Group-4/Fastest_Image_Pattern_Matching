
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

	void MatchTemplateBase (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, BOOL bUseSIMD);

	void MatchTemplateChanged(cv::Mat &matSrc, s_TemplData *pTemplData, cv::Mat &matResult, int iLayer,
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

		MatchTemplateBase (matRotatedSrc, pTemplData, matResult, iTopLayer, FALSE);

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
					MatchTemplateBase (matRotatedSrc, pTemplData, matResult, iLayer, TRUE);
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
		printf("%02hhd ", Y);
	}
	printf("\n");

	for (int Y = 0; Y < 32; Y++) {
		printf("%02hhx ", ((char *)vec)[Y]);
	}
	printf("\n");
	printf("\n");
}

void printBytes128(__m128i* vec) {
	for (int Y = 0; Y < 16; Y++) {
		printf("%02hhd ", Y);
	}
	printf("\n");

	for (int Y = 0; Y < 16; Y++) {
		printf("%02hhx ", ((char *)vec)[Y]);
	}
	printf("\n");
	printf("\n");
}

static const char iotaArr64[64] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };
static const __m256i iota64 = *(__m256i*)&iotaArr64;
static const int iotaArr8[8] = {0,1,2,3,4,5,6,7};
static const __m256i iota8 = *(__m256i*)&iotaArr8;

//
inline int IM_Conv_SIMD_AVX (unsigned char* pCharKernel, unsigned char *pCharConv, int iLength)
{
	const int iBlockSize = 32, Block = iLength / iBlockSize;
	__m256i SumV = _mm256_setzero_si256 ();
	__m256i Zero = _mm256_setzero_si256 ();



	for (int Y = 0; Y < iLength; Y += iBlockSize)
	{
		char diff = min(iLength - Y, 127);
		__m256i clownMask = _mm256_set1_epi8(diff);
		clownMask = _mm256_sub_epi8(clownMask, iota64);

		clownMask = _mm256_cmpgt_epi8( clownMask, Zero);

		__m256i SrcK = _mm256_loadu_si256((__m256i*)(pCharKernel + Y));
		__m256i SrcC = _mm256_loadu_si256 ((__m256i*)(pCharConv + Y));

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

// Utility
template<typename  T>
void printMat(cv::Mat mat) {
	for (int Y = 0; Y < mat.rows; Y++) {
		for (int X = 0; X < mat.cols; X++) {
			std::cout << mat.at<T>(Y, X) << " ";
		}
		printf("\n");
	}
}

// Helpers for masking
// this creates a mask for the first n bytes
inline __m128i mask_shift_2(uint32_t n)
{
	static const int8_t mask_lut[32] = {
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	  };
	return _mm_loadu_si128((__m128i *)(mask_lut + 16 - n));
}


inline __m256i mask_shift_256(uint32_t n)
{
	static const int8_t mask_lut[64] = {
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	  };
	return _mm256_loadu_si256((__m256i *)(mask_lut + 32 - n));
}

const auto vec_zero = _mm256_setzero_si256();
const auto vec_one = _mm256_set1_epi32(1);


void CMatchToolDlg::MatchTemplateBase (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, BOOL bUseSIMD)
{
	if (m_ckSIMD && bUseSIMD)
	{
		// printf("SIMD\n");
		//From ImageShop
		cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];
		matResult.create (matSrc.rows - matTemplate.rows + 1,matSrc.cols - matTemplate.cols + 1, CV_32FC1);
		matResult.setTo (0);

// uncomment the line to test different parallelization tactics
//#define MATCH_TEMPLATE_BASE
//#define MATCH_TEMPLATE_SEQUENTIAL_SOURCE_TEMPLATE
//#define MATCH_TEMPLATE_SEQUENTIAL_TEMPLATE_SOURCE
//#define MATCH_TEMPLATE_SEQUENTIAL_RESULT_TEMPLATE
//#define MATCH_TEMPLATE_LOAD_TO_RESULT_BATCH_4
//#define MATH_TEMLATED_LOAD_TO_RESULT_BATCH_16
//#define MATCH_TEMPLATE_LOAD_TO_RESULT_BATCH_24
//#define MATCH_TEMPLATE_SIMD
//#define MATCH_TEMPLATE_OPENCV


#if defined(MATCH_TEMPLATE_BASE)
		// this is the original loop convolution that was used by the author
		// code is left unchanged it already has some SIMD parallelisation
		// using SSE 128-bit registers
        // In this version we already loop over the
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
				for (t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols)
				{
					*r_matResult = *r_matResult + IM_Conv_SIMD (r_template, r_sub_source, matTemplate.cols);
				}
			}
		}


#elif defined(MATCH_TEMPLATE_SEQUENTIAL_SOURCE_TEMPLATE)
		// An experimental version where we try to first loop over the "source" matrix and then we over the template.
		// It was tricky to the proper indexing
        // Also really slow

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
#elif defined(MATCH_TEMPLATE_SEQUENTIAL_TEMPLATE_SOURCE)
		//  This version is the same as the previous one but we first iterate over the template matrix and
        // then over the source. This is about 18% faster
		for (int tr = 0; tr < matTemplate.rows; tr++) {
			uchar* r_templ = matTemplate.ptr<uchar> (tr);
			for (int sr = max(0, tr); sr < matResult.rows + tr && sr < matSrc.rows; sr++) {
				uchar* r_src = matSrc.ptr<uchar> (sr);
				int rr = sr - tr;
				float* r_res = matResult.ptr<float> (rr);
				for (int tc = 0; tc < matTemplate.cols; tc++) {
					int batchSize = 1;
					for (int sc = max(0,  tc); sc < matResult.cols + tc && sc < matSrc.cols; sc+=batchSize) {
						int rc = sc - tc;
						r_res[rc] += r_templ[tc] * r_src[sc];
					}
				}
			}
		}
#elif defined(MATCH_TEMPLATE_SEQUENTIAL_RESULT_TEMPLATE)
        // In this version we first iterate over the result matrix and then over the template matrix
        // the whole point of trying these iteration orders is to see if it affects performance (because of cpu caching)
        // Also different iteration orders will allow us to do different types of parallelizations
		for (int rr = 0; rr < matResult.rows; rr++) {
			auto* r_res = matResult.ptr<float> (rr);
			for (int rc = 0; rc < matResult.cols; rc++) {
				for (int tr = 0; tr < matTemplate.rows; tr++) {
					auto* r_src = matSrc.ptr<uchar>(rr + tr);
					auto* r_templ = matTemplate.ptr<uchar>(tr);

					for (int tc = 0; tc < matTemplate.cols; tc++) {
						r_res[rc] += r_templ[tc] * r_src[rc + tc];
					}
				}

			}
		}
#elif defined(MATCH_TEMPLATE_LOAD_TO_RESULT_BATCH_4)
        // This is the parallel version of a loop where we first iterate over the template matrix and then iterate over
		// the result matrix this allows us to write to 8 float to the result matrix at a time:
        // Here is a picture of the access pattern:

//            [tmp 0][tmp 1][tmp 0][tmp 1][tmp 0][tmp 1][tmp 0][tmp 1][tmp 0][tmp 1][tmp 0][tmp 1][tmp 0][tmp 1][tmp 0][tmp 1]
//               *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *
//            [src 0][src 1][src 1][src 2][src 2][src 3][src 3][src 4][src 4][src 5][src 5][src 6][src 6][src 7][src 7][src 8]
//                |     |       |     |       |     |       |     |       |     |       |     |       |     |       |     |
//                |--+--|       |--+--|       |--+--|       |--+--|       |--+--|       |--+--|       |--+--|       |--+--|
//                   |             |             |             |             |             |             |             |
//            [     r0     ][     r1     ][     r2     ][     r3     ][     r4     ][     r5     ][     r6     ][     r7     ]

        // Notice the interleaving pattern for the source values. We need to interleave them due to the way convolution works
        // Actually we repeat that one more time with the source columns shifted over 2 and interleaved one more time
        // Doing it twice will save us some loads and conversions to floats which are pretty expensive

		for(int tr = 0; tr < matTemplate.rows; tr++) {
			uchar* r_templ = matTemplate.ptr<uchar> (tr);
			for (int rr = 0; rr < matResult.cols; rr++) {
				float* r_res = matResult.ptr<float> (rr);
				uchar* r_src = matSrc.ptr<uchar> (tr + rr);

				int bigBlockSize = 4;
				for (int tc = 0; tc < matTemplate.cols; tc+=bigBlockSize) {
					int blockSize = 8;

                    // Load in the template matrix
					auto tmp = _mm_cvtepu8_epi16(_mm_loadu_epi8(r_templ + tc));
					// Mask it so we do not write over the batch size
                    int shift = max(0, tc + 4 - matTemplate.cols);
					tmp = _mm_slli_epi64(tmp, shift*16);
					tmp = _mm_srli_epi64(tmp, shift*16);

					// Repeat the first 2 numbers over the whole register
					auto vec_t1 = _mm256_broadcastd_epi32(tmp);
                    // Get another register where we repeat numbers 3 - 4
					auto vec_t2 = _mm256_broadcastd_epi32(
						_mm_srli_si128(tmp, 4)
					);

					for (int rc = 0; rc < matResult.cols; rc+=blockSize) {
						// we load in 16
						// But we only use 11
						// These are the first interealed sections of source values
						auto shifted_0 =  _mm_loadu_si128((__m128i*)(r_src + tc + rc));
						auto shifted_1 = _mm_srli_si128(shifted_0, 1);
						auto interleaved_1 = _mm_unpacklo_epi16(
							(shifted_0),
							shifted_1
						);

						auto shifted_2 = _mm_srli_si128(shifted_0, 2);
						auto shifted_3 = _mm_srli_si128(shifted_0, 3);
						auto interleaved_2 = _mm_unpacklo_epi16(
							(shifted_2),
							shifted_3
						);

						auto extended_1 = _mm256_cvtepu8_epi16(interleaved_1);
						auto extended_2 = _mm256_cvtepu8_epi16(interleaved_2);
						__m256i vec_mul1 = _mm256_madd_epi16(vec_t1, extended_1);
						__m256i vec_mul2 = _mm256_madd_epi16(vec_t2, extended_2);
						__m256 vec_mul = _mm256_cvtepi32_ps(_mm256_add_epi32(vec_mul1, vec_mul2));

						auto vec_r = _mm256_loadu_ps(r_res + rc);
						vec_r = _mm256_add_ps(vec_r, vec_mul);
						__m256i mask = _mm256_set1_epi32 (matResult.cols - rc );
						mask = _mm256_sub_epi32 (mask, iota8);
						mask = _mm256_cmpgt_epi32(mask, vec_zero);
						_mm256_maskstore_ps(r_res+rc, mask, vec_r);
					}
				}
			}
		}
// Below you can find the same versions as above but scaled to 8, 16 and 24 batchsize for the values loaded from the template
// Code is pretty simillar execpt we had to do some extra work around the fact that the 256 registers actually have lanes
#elif defined(MATCH_TEMPLATE_LOAD_TO_RESULT_BATCH_8)
		for(int tr = 0; tr < matTemplate.rows; tr++) {
			uchar* r_templ = matTemplate.ptr<uchar> (tr);
			for (int rr = 0; rr < matResult.cols; rr++) {
				float* r_res = matResult.ptr<float> (rr);
				uchar* r_src = matSrc.ptr<uchar> (tr + rr);

				int bigBlockSize = 8;
				int blocks = matTemplate.cols / bigBlockSize;
				for (int tc = 0; tc < matTemplate.cols; tc+=bigBlockSize) {

					auto tmp = _mm_cvtepu8_epi16(_mm_loadu_epi8(r_templ + tc));
					auto to_read = min(matTemplate.cols - tc, bigBlockSize);
					auto mask = mask_shift_2( to_read * 2);
					// printBytes128(&mask);
					tmp = _mm_and_si128(mask, tmp);

					auto vec_t1 = _mm256_broadcastd_epi32(tmp);
					auto vec_t2 = _mm256_broadcastd_epi32(_mm_srli_si128(tmp, 4));
					auto vec_t3 = _mm256_broadcastd_epi32(_mm_srli_si128(tmp, 8));
					auto vec_t4 = _mm256_broadcastd_epi32(_mm_srli_si128(tmp, 12));

					int blockSize = 8;
					for (int rc = 0; rc < matResult.cols; rc+=blockSize) {
						// we load in 16
						// But we only use 15
						auto shifted_0 =  _mm_loadu_si128((__m128i*)(r_src + tc + rc));
						auto shifted_1 = _mm_srli_si128(shifted_0, 1);
						auto shifted_2 = _mm_srli_si128(shifted_0, 2);
						auto shifted_3 = _mm_srli_si128(shifted_0, 3);
						auto shifted_4 = _mm_srli_si128(shifted_0, 4);
						auto shifted_5 = _mm_srli_si128(shifted_0, 5);
						auto shifted_6 = _mm_srli_si128(shifted_0, 6);
						auto shifted_7 = _mm_srli_si128(shifted_0, 7);

						// this is 128 bits long and it spans over 8 bits
						// 00 01 01 02 02 03 03 04 04 05 05 06 06 07 07 08
						// 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02
						// 00 00 00 01 00 01 00 02 00 02 00 03 00 03 00 04 00 04 00 05 00 05 00 06 00 06 00 07 00 07 00 08
						auto interleaved_1 = _mm_unpacklo_epi16(shifted_0,shifted_1);
						auto interleaved_2 = _mm_unpacklo_epi16(shifted_2,shifted_3);
						auto interleaved_3 = _mm_unpacklo_epi16(shifted_4,shifted_5);
						auto interleaved_4 = _mm_unpacklo_epi16(shifted_6,shifted_7);

						auto extended_1 = _mm256_cvtepu8_epi16(interleaved_1);
						auto extended_2 = _mm256_cvtepu8_epi16(interleaved_2);
						auto extended_3 = _mm256_cvtepu8_epi16(interleaved_3);
						auto extended_4 = _mm256_cvtepu8_epi16(interleaved_4);

						__m256i vec_mul1 = _mm256_madd_epi16(vec_t1, extended_1);
						__m256i vec_mul2 = _mm256_madd_epi16(vec_t2, extended_2);
						__m256i vec_mul3 = _mm256_madd_epi16(vec_t3, extended_3);
						__m256i vec_mul4 = _mm256_madd_epi16(vec_t4, extended_4);

						__m256i sum1 = _mm256_add_epi32(vec_mul1, vec_mul2);
						__m256i sum2 = _mm256_add_epi32(vec_mul3, vec_mul4);

						__m256i sum = _mm256_add_epi32(sum1, sum2);
						__m256 vec_mul = _mm256_cvtepi32_ps(sum);

						auto vec_r = _mm256_loadu_ps(r_res + rc);
						vec_r = _mm256_add_ps(vec_r, vec_mul);
						__m256i mask = _mm256_set1_epi32 (matResult.cols - rc );
						mask = _mm256_sub_epi32 (mask, iota8);
						mask = _mm256_cmpgt_epi32(mask, vec_zero);
						_mm256_maskstore_ps(r_res+rc, mask, vec_r);
					}
				}
			}
		}
#elif defined(MATH_TEMLATED_LOAD_TO_RESULT_BATCH_16)
			for(int tr = 0; tr < matTemplate.rows; tr++) {
				uchar* r_templ = matTemplate.ptr<uchar> (tr);
				for (int rr = 0; rr < matResult.cols; rr++) {
					float* r_res = matResult.ptr<float> (rr);
					uchar* r_src = matSrc.ptr<uchar> (tr + rr);

					const int bigBlockSize = 16;
					int blocks = matTemplate.cols / bigBlockSize;
					for (int tc = 0; tc < matTemplate.cols; tc+=bigBlockSize) {

						auto tmp = _mm256_cvtepu8_epi16(_mm_loadu_epi8(r_templ + tc));
						auto to_read = min(matTemplate.cols - tc, bigBlockSize);
						auto mask = mask_shift_256( to_read * 2);
						tmp = _mm256_and_si256(mask, tmp);

						__m256i vec_t[8];
						auto vec_i = vec_zero;
						for (int i = 0; i < 8; i++) {
							// This is kind of expnsive, but it still cheaper than broadcasting
							vec_t[i] = _mm256_permutevar8x32_epi32(tmp, vec_i);

							vec_i = _mm256_add_epi32(vec_i, vec_one);
						}

						int blockSize = 8;
						for (int rc = 0; rc < matResult.cols; rc+=blockSize) {
							// we load in 32
							// But we only use 8 + 15 = 23
							auto vec_src1 =  _mm_loadu_si128((__m128i*)(r_src + tc + rc));
							auto vec_src2 =  _mm_loadu_si128((__m128i*)(r_src + tc + rc + 8));

							auto shifted_0 = vec_src1;
							auto shifted_1 = _mm_srli_si128(shifted_0, 1);
							auto shifted_2 = _mm_srli_si128(shifted_0, 2);
							auto shifted_3 = _mm_srli_si128(shifted_0, 3);
							auto shifted_4 = _mm_srli_si128(shifted_0, 4);
							auto shifted_5 = _mm_srli_si128(shifted_0, 5);
							auto shifted_6 = _mm_srli_si128(shifted_0, 6);
							auto shifted_7 = _mm_srli_si128(shifted_0, 7);

							auto shifted_8= vec_src2;
							auto shifted_9 = _mm_srli_si128(shifted_8, 1);
							auto shifted_10 = _mm_srli_si128(shifted_8, 2);
							auto shifted_11 = _mm_srli_si128(shifted_8, 3);
							auto shifted_12 = _mm_srli_si128(shifted_8, 4);
							auto shifted_13 = _mm_srli_si128(shifted_8, 5);
							auto shifted_14 = _mm_srli_si128(shifted_8, 6);
							auto shifted_15 = _mm_srli_si128(shifted_8, 7);

							// this is 128 bits long and it spans over 8 bits
							// 00 01 01 02 02 03 03 04 04 05 05 06 06 07 07 08
							// 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02
							// 00 00 00 01 00 01 00 02 00 02 00 03 00 03 00 04 00 04 00 05 00 05 00 06 00 06 00 07 00 07 00 08
							auto interleaved_1 = _mm_unpacklo_epi16(shifted_0,shifted_1);
							auto interleaved_2 = _mm_unpacklo_epi16(shifted_2,shifted_3);
							auto interleaved_3 = _mm_unpacklo_epi16(shifted_4,shifted_5);
							auto interleaved_4 = _mm_unpacklo_epi16(shifted_6,shifted_7);
							auto interleaved_5 = _mm_unpacklo_epi16(shifted_8,shifted_9);
							auto interleaved_6 = _mm_unpacklo_epi16(shifted_10,shifted_11);
							auto interleaved_7 = _mm_unpacklo_epi16(shifted_12,shifted_13);
							auto interleaved_8 = _mm_unpacklo_epi16(shifted_14,shifted_15);

							auto extended_1 = _mm256_cvtepu8_epi16(interleaved_1);
							auto extended_2 = _mm256_cvtepu8_epi16(interleaved_2);
							auto extended_3 = _mm256_cvtepu8_epi16(interleaved_3);
							auto extended_4 = _mm256_cvtepu8_epi16(interleaved_4);
							auto extended_5 = _mm256_cvtepu8_epi16(interleaved_5);
							auto extended_6 = _mm256_cvtepu8_epi16(interleaved_6);
							auto extended_7 = _mm256_cvtepu8_epi16(interleaved_7);
							auto extended_8 = _mm256_cvtepu8_epi16(interleaved_8);

							__m256i vec_mul1 = _mm256_madd_epi16(vec_t[0], extended_1);
							__m256i vec_mul2 = _mm256_madd_epi16(vec_t[1], extended_2);
							__m256i vec_mul3 = _mm256_madd_epi16(vec_t[2], extended_3);
							__m256i vec_mul4 = _mm256_madd_epi16(vec_t[3], extended_4);
							__m256i vec_mul5 = _mm256_madd_epi16(vec_t[4], extended_5);
							__m256i vec_mul6 = _mm256_madd_epi16(vec_t[5], extended_6);
							__m256i vec_mul7 = _mm256_madd_epi16(vec_t[6], extended_7);
							__m256i vec_mul8 = _mm256_madd_epi16(vec_t[7], extended_8);

							__m256i sum1 = _mm256_add_epi32(vec_mul1, vec_mul2);
							__m256i sum2 = _mm256_add_epi32(vec_mul3, vec_mul4);
							__m256i sum3 = _mm256_add_epi32(vec_mul5, vec_mul6);
							__m256i sum4 = _mm256_add_epi32(vec_mul7, vec_mul8);

							__m256i sum5 = _mm256_add_epi32(sum1, sum2);
							__m256i sum6 = _mm256_add_epi32(sum3, sum4);
							__m256i sum = _mm256_add_epi32(sum5, sum6);
							__m256 vec_mul = _mm256_cvtepi32_ps(sum);

							auto vec_r = _mm256_loadu_ps(r_res + rc);
							vec_r = _mm256_add_ps(vec_r, vec_mul);
							__m256i mask = _mm256_set1_epi32 (matResult.cols - rc );
							mask = _mm256_sub_epi32 (mask, iota8);
							mask = _mm256_cmpgt_epi32(mask, vec_zero);
							_mm256_maskstore_ps(r_res+rc, mask, vec_r);
						}
					}
				}
			}
#elif defined(MATCH_TEMPLATE_LOAD_TO_RESULT_BATCH_24)
			for(int tr = 0; tr < matTemplate.rows; tr++) {
				uchar* r_templ = matTemplate.ptr<uchar> (tr);
				for (int rr = 0; rr < matResult.cols; rr++) {
					float* r_res = matResult.ptr<float> (rr);
					uchar* r_src = matSrc.ptr<uchar> (tr + rr);

					const int bigBlockSize = 24;
					int blocks = matTemplate.cols / bigBlockSize;
					for (int tc = 0; tc < matTemplate.cols; tc+=bigBlockSize) {
						auto to_read = min(matTemplate.cols - tc, bigBlockSize);
						auto mask = mask_shift_256( to_read  );
						auto tmp = _mm256_loadu_epi8(r_templ + tc);
						tmp = _mm256_and_si256(tmp, mask);

						auto tmp_lo = _mm256_unpacklo_epi8(tmp, vec_zero);
						auto tmp_hi = _mm256_unpackhi_epi8(tmp, vec_zero);

						__m256i vec_t2[16];
						auto vec_i = vec_zero;
						for (int i = 0; i <4; i++) {
							vec_t2[i] = _mm256_permutevar8x32_epi32(tmp_lo, vec_i);
							vec_i = _mm256_add_epi32(vec_i, vec_one);
						}
						for (int i = 8; i <12; i++) {
							vec_t2[i] = _mm256_permutevar8x32_epi32(tmp_lo, vec_i);
							vec_i = _mm256_add_epi32(vec_i, vec_one);
						}

						vec_i = vec_zero;
						for (int i = 4; i <8; i++) {
							vec_t2[i] = _mm256_permutevar8x32_epi32(tmp_hi, vec_i);
							vec_i = _mm256_add_epi32(vec_i, vec_one);
						}
						// for (int i = 12; i <16; i++) {
							// vec_t2[i] = _mm256_permutevar8x32_epi32(tmp_hi, vec_i);
							// vec_i = _mm256_add_epi32(vec_i, vec_one);
						// }


						int blockSize = 8;
						for (int rc = 0; rc < matResult.cols; rc+=blockSize) {
							// we load in 32
							// But we only use 8 + 23 = 31
							__m128i vec_src[3];
							vec_src[0] =  _mm_loadu_si128((__m128i*)(r_src + tc + rc));
							vec_src[1] =  _mm_loadu_si128((__m128i*)(r_src + tc + rc + 8));
							vec_src[2] =  _mm_loadu_si128((__m128i*)(r_src + tc + rc + 16));
							// vec_src[3] =  _mm_loadu_si128((__m128i*)(r_src + tc + rc + 24));

							int srcBlockSize = 24;

							__m128i shifted[srcBlockSize];
							for (int i = 0; i <srcBlockSize/8; i++) {
								shifted[i*8+0] = vec_src[i];
								shifted[i*8+1] = _mm_srli_si128(shifted[i*8], 1);
								shifted[i*8+2] = _mm_srli_si128(shifted[i*8], 2);
								shifted[i*8+3] = _mm_srli_si128(shifted[i*8], 3);
								shifted[i*8+4] = _mm_srli_si128(shifted[i*8], 4);
								shifted[i*8+5] = _mm_srli_si128(shifted[i*8], 5);
								shifted[i*8+6] = _mm_srli_si128(shifted[i*8], 6);
								shifted[i*8+7] = _mm_srli_si128(shifted[i*8], 7);
							}

							// this is 128 bits long and it spans over 8 bits
							// 00 01 01 02 02 03 03 04 04 05 05 06 06 07 07 08
							// 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02 00 01 00 02
							// 00 00 00 01 00 01 00 02 00 02 00 03 00 03 00 04 00 04 00 05 00 05 00 06 00 06 00 07 00 07 00 08
							__m256i vec_mul[srcBlockSize / 2];
							for (int i = 0; i < srcBlockSize / 2; i++) {
								vec_mul[i] = _mm256_madd_epi16(vec_t2[i],  _mm256_cvtepu8_epi16(_mm_unpacklo_epi16(shifted[i*2], shifted[i*2+1])));
							}

							__m256i sum1 = _mm256_add_epi32(vec_mul[0] , vec_mul[1]);
							__m256i sum2 = _mm256_add_epi32(vec_mul[2], vec_mul[3]);
							__m256i sum3 = _mm256_add_epi32(vec_mul[4], vec_mul[5]);
							__m256i sum4 = _mm256_add_epi32(vec_mul[6], vec_mul[7]);
							__m256i sum5 = _mm256_add_epi32(vec_mul[8] , vec_mul[9]);
							__m256i sum6 = _mm256_add_epi32(vec_mul[10], vec_mul[11]);

							__m256i sum9 = _mm256_add_epi32(sum1, sum2);
							__m256i sum10 = _mm256_add_epi32(sum3, sum4);
							__m256i sum11 = _mm256_add_epi32(sum5, sum6);

							__m256i sum13 = _mm256_add_epi32(sum9, sum10);


							__m256i sum = _mm256_add_epi32(sum13, sum11);
							__m256 vec_sum_f = _mm256_cvtepi32_ps(sum);

							auto vec_r = _mm256_loadu_ps(r_res + rc);
							vec_r = _mm256_add_ps(vec_r, vec_sum_f);
							__m256i mask = _mm256_set1_epi32 (matResult.cols - rc );
							mask = _mm256_sub_epi32 (mask, iota8);
							mask = _mm256_cmpgt_epi32(mask, vec_zero);
							_mm256_maskstore_ps(r_res+rc, mask, vec_r);
						}
					}
				}
			}
#elif defined(MATCH_TEMPLATE_SIMD)
		// Just like in the loops above we iterate over the template and then over the result, in this loop we have the same pattern
        // but this time the parallelization is as follows:

//		        [src 0][src 1][src 2][src 3][src 4][src 5][src 6][src 7][src 8][src 9][src10][src11][src12][src13][src14][src15]
//			       *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *
//	        	[tmp 0][tmp 1][tmp 2][tmp 3][tmp 4][tmp 5][tmp 6][tmp 7][tmp 8][tmp 9][tmp10][tmp11][tmp12][tmp13][tmp14][tmp15]
//                   |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
//		result =   +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +

//     So we do a vertical multiplication and a horizontal addition. We also do some expanding into int16 from bytes since AVX does
//     not support multiplication of uchars :(
		for(int tr = 0; tr < matTemplate.rows; tr++) {
			uchar* r_templ = matTemplate.ptr<uchar> (tr);
			for (int rr = 0; rr < matResult.cols; rr++) {
				float* r_res = matResult.ptr<float> (rr);
				uchar* r_src = matSrc.ptr<uchar> (tr + rr);

				int templ_block_size = 32;
				const auto vec_zero = _mm256_setzero_si256();
				auto vec_sum = _mm256_setzero_si256();

				for (int tc = 0; tc < matTemplate.cols; tc+=templ_block_size) {
					__m256i mask = _mm256_set1_epi8 (min( matTemplate.cols - tc,127));
					mask = _mm256_sub_epi8 (mask, iota64);
					mask = _mm256_cmpgt_epi8 (mask, vec_zero );

					__m256i vec_templ = _mm256_loadu_si256 ((__m256i*)(r_templ + tc));
					vec_templ = _mm256_and_si256 (vec_templ, mask);
					__m256i vec_templ_lo = _mm256_unpacklo_epi8 (vec_templ, vec_zero);
					__m256i vec_templ_hi = _mm256_unpackhi_epi8 (vec_templ, vec_zero);
					for (int rc = 0; rc < matResult.cols; rc++) {

						__m256i vec_src1 = _mm256_loadu_si256 ((__m256i*)(r_src + tc + rc));
						__m256i vec_src_lo = _mm256_unpacklo_epi8 (vec_src1, vec_zero);
						__m256i vec_src_hi = _mm256_unpackhi_epi8 (vec_src1, vec_zero);

						__m256i vec_tmp_sum = _mm256_add_epi32 (_mm256_madd_epi16 (vec_src_lo, vec_templ_lo), _mm256_madd_epi16 (vec_src_hi, vec_templ_hi));
						r_res[rc] += _mm256_hsum_epi32 (vec_tmp_sum);
					}
				}
			}
		}
#elif defined(MATCH_TEMPLATE_OPENCV)
	matchTemplate (matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);
#endif


	}
	else
		matchTemplate (matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);

#define COEFF_DENOMINATOR_SIMD
#if defined(COEFF_DENOMINATOR_SIMD)
	CCOEFF_Denominator_SIMD_AVX (matSrc, pTemplData, matResult, iLayer);
#else
    CCOEFF_Denominator (matSrc, pTemplData, matResult, iLayer);
#endif
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

// Original function
void CMatchToolDlg::CCOEFF_Denominator (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
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

	{
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

				rrow[j] = (float)num;
			}
		}
	}
}


// SIMD Version
inline void CMatchToolDlg::CCOEFF_Denominator_SIMD_AVX (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
{
	// This is a parallelized version of the above function.
    // In some comments I have put in the original code

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

	double dTemplMean = pTemplData->vecTemplMean[iLayer][0];
	double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
	double dInvArea = pTemplData->vecInvArea[iLayer];

	// some constants that we are going to use
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
		for (int i = 0; i < matResult.rows; i++)
		{
			auto* rrow = matResult.ptr<float> (i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			// We can load 4 doubles in 256 bit regsiters
			int blockSize = 4;
			for (int j=0; j < matResult.cols; j+=blockSize)
			{
				// we now load 4 doubles from the p1 array
				// double t = p0[idx + j] - p1[idx + j] - p2[idx + j] + p3[idx + j];
				__m256d vec_p0 = _mm256_loadu_pd(&p0[idx +j]);
				__m256d vec_p1 = _mm256_loadu_pd(&p1[idx +j]);
				__m256d vec_p2 = _mm256_loadu_pd(&p2[idx +j]);
				__m256d vec_p3 = _mm256_loadu_pd(&p3[idx +j]);
				__m256d t = _mm256_add_pd(vec_p0, vec_p3);
				t = _mm256_sub_pd(t, vec_p1);
				t = _mm256_sub_pd(t, vec_p2);

				// double num = rrow[j] - t * dTemplMean;
				__m128 numFloats = _mm_loadu_ps(&rrow[j]);
				__m256d vec_num = _mm256_cvtps_pd(numFloats); // Expensive cast to doubles... can we prevent that somehow?
				vec_num = _mm256_fnmadd_pd(t, vec_dTemplMean, vec_num);

				// double wndMean = t * t * dInvArea;
				__m256d vec_wndMean = _mm256_mul_pd(_mm256_mul_pd(t, t), vec_dInvArea);

				// double wndSum2 = q0[idx2 + j] - q1[idx + j] - q2[idx2 + j] + q3[idx2 + j];
				__m256d vec_q0 = _mm256_loadu_pd(&q0[idx2 +j]);
				__m256d vec_q1 = _mm256_loadu_pd(&q1[idx2 +j]);
				__m256d vec_q2 = _mm256_loadu_pd(&q2[idx2 +j]);
				__m256d vec_q3 = _mm256_loadu_pd(&q3[idx2 +j]);
				__m256d vec_wndSum2 = _mm256_add_pd(vec_q0, vec_q3);
				vec_wndSum2 = _mm256_sub_pd(vec_wndSum2, vec_q1);
				vec_wndSum2 = _mm256_sub_pd(vec_wndSum2, vec_q2);

				// double diff = MAX (wndSum2 - wndMean, 0);
				__m256d vec_diff = _mm256_max_pd(_mm256_sub_pd(vec_wndSum2, vec_wndMean), zeros);

                // std::min (0.5, 10 * FLT_EPSILON * wndSum2)
				__m256d vec_diffThresh = _mm256_min_pd(halves, _mm256_mul_pd( vec_wndSum2,epsilon ));

				// double threshold = 0;
				// if (diff <= std::min (0.5, 10 * FLT_EPSILON * wndSum2))
				// 	threshold = 0; // avoid rounding errors
				// else
				// 	threshold = std::sqrt (diff)*dTemplNorm;
                // We use masking with andnot to emulate the if case scenario
				__m256d threshold_val_false = _mm256_mul_pd(_mm256_sqrt_pd(vec_diff), _mm256_set1_pd(dTemplNorm));
				__m256d vec_threshold_mask = _mm256_cmp_pd(vec_diff, vec_diffThresh, _CMP_LE_OS);
				__m256d vec_threshold = _mm256_andnot_pd(vec_threshold_mask, threshold_val_false);


				// fabs (num)
				// ( from https://stackoverflow.com/questions/5508628/how-to-absolute-2-double-or-4-floats-using-sse-instruction-set-up-to-sse4)
				__m256d sign_mask = _mm256_set1_pd(-0.f);
				__m256d vec_num_abs = _mm256_andnot_pd(sign_mask, vec_num);

				// We precompute some masks for the sequence of ifs below
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
				__m128 result = _mm256_cvtpd_ps(vec_num);

				// This is to prevent overflows when the block size does not divide
				// for example we have cols = 19, blocksize = 4
				// So blocks = 4
				// At the last iterations we have j = 4, and blocksize = 4
				// So we are going to write at adresses j*4, j*4+1 ,j*4+2, j*4+3,
				// which is not good, since addr j*4+3 is out of bounds
				// To fix that we need to mask out any loads that happen outside the bounds
				__m128i writeMask = _mm_set1_epi32(matResult.cols - j); // -1 is important
				writeMask = _mm_sub_epi32(writeMask, iota);
				writeMask = _mm_cmplt_epi32(zerosi, writeMask);

				_mm_maskstore_ps(&rrow[j], writeMask, result);
			}
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