import { useLocation, useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, CheckCircle, XCircle, AlertTriangle, Building } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const LoanResults = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, formData } = location.state || {};

  if (!result) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-violet-100 flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="text-center py-8">
            <p>No results found. Please go back and try again.</p>
            <Button onClick={() => navigate('/loan-application')} className="mt-4">
              Back to Loan Application
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const getDecisionIcon = (decision: string) => {
    switch (decision) {
      case "APPROVED":
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case "REJECTED":
        return <XCircle className="w-6 h-6 text-red-500" />;
      default:
        return <AlertTriangle className="w-6 h-6 text-yellow-500" />;
    }
  };

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case "APPROVED":
        return "default";
      case "REJECTED":
        return "destructive";
      default:
        return "secondary";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-violet-100">
      <div className="container mx-auto px-4 py-8">
        <Button 
          onClick={() => navigate('/loan-application')} 
          variant="ghost" 
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Loan Application
        </Button>

        <div className="max-w-6xl mx-auto space-y-6">
          <Card className="shadow-lg border-0">
            <CardHeader className="text-center">
              <div className="w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Building className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-2xl font-bold">Loan Application Results</CardTitle>
              <CardDescription>AI-powered loan processing complete</CardDescription>
            </CardHeader>
          </Card>

          <Card className="shadow-lg border-0">
            <CardHeader className="text-center">
              <div className="flex items-center justify-center gap-3 mb-4">
                {getDecisionIcon(result.final_decision)}
                <span className="text-2xl font-bold">
                  {result.final_decision}
                </span>
              </div>
              <Badge variant={getDecisionColor(result.final_decision)} className="text-lg px-4 py-2">
                {result.final_decision}
              </Badge>
            </CardHeader>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="text-lg">Rule-Based Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Decision:</span>
                    <Badge variant={result.rule_based.approved ? "default" : "destructive"}>
                      {result.rule_based.approved ? "APPROVED" : "REJECTED"}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Rules Passed:</span>
                    <span className="text-sm font-medium">
                      {result.rule_based.rules_passed}/{result.rule_based.total_rules}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Score:</span>
                    <span className="text-sm font-medium">{result.rule_based.rule_score}/100</span>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="text-lg">ML Prediction</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Prediction:</span>
                    <Badge variant={result.ml_prediction.prediction === "APPROVED" ? "default" : "destructive"}>
                      {result.ml_prediction.prediction}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Confidence:</span>
                    <span className="text-sm font-medium">
                      {(result.ml_prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${result.ml_prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="text-lg">AI Suggestions</CardTitle>
              <CardDescription>Recommendations from Groq AI</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-50 p-4 rounded-lg border">
                <p className="text-sm whitespace-pre-wrap">
                  {result.groq_suggestion.suggestion}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="text-lg">Application Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Loan Amount:</span>
                  <p className="font-medium">${parseInt(formData.loan_amount).toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-gray-600">Annual Income:</span>
                  <p className="font-medium">${parseInt(formData.income_annum).toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-gray-600">CIBIL Score:</span>
                  <p className="font-medium">{formData.cibil_score}</p>
                </div>
                <div>
                  <span className="text-gray-600">Loan Term:</span>
                  <p className="font-medium">{formData.loan_term} months</p>
                </div>
                <div>
                  <span className="text-gray-600">Dependents:</span>
                  <p className="font-medium">{formData.no_of_dependents}</p>
                </div>
                <div>
                  <span className="text-gray-600">Verification Score:</span>
                  <p className="font-medium">{result.verification_score}/100</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-center space-x-4">
            <Button 
              onClick={() => navigate('/loan-application')}
              className="bg-purple-500 hover:bg-purple-600"
            >
              Submit New Application
            </Button>
            <Button variant="outline" onClick={() => navigate('/')}>
              Back to Home
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoanResults;
