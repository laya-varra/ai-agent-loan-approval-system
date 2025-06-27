
import { useLocation, useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, CheckCircle, XCircle, CreditCard } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const PanResults = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, fileName } = location.state || {};

  if (!result) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="text-center py-8">
            <p>No results found. Please go back and try again.</p>
            <Button onClick={() => navigate('/pan-verification')} className="mt-4">
              Back to PAN Verification
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <Button 
          onClick={() => navigate('/pan-verification')} 
          variant="ghost" 
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to PAN Verification
        </Button>

        <div className="max-w-4xl mx-auto space-y-6">
          <Card className="shadow-lg border-0">
            <CardHeader className="text-center">
              <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <CreditCard className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-2xl font-bold">PAN Card Extraction Results</CardTitle>
              <CardDescription>File: {fileName}</CardDescription>
            </CardHeader>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  {result.valid ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-500" />
                  )}
                  Validation Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <Badge variant={result.valid ? "default" : "destructive"}>
                      {result.valid ? "Valid PAN Format" : "Invalid PAN Format"}
                    </Badge>
                  </div>
                  {result.pan_number && (
                    <div>
                      <h4 className="font-semibold text-sm text-gray-600 mb-1">Extracted PAN Number</h4>
                      <p className="text-lg font-mono bg-gray-50 p-2 rounded border">
                        {result.pan_number}
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="text-lg">Processing Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Status:</span>
                    <Badge variant={result.valid ? "default" : "secondary"}>
                      {result.valid ? "Success" : "Needs Review"}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">PAN Detected:</span>
                    <span className="text-sm font-medium">
                      {result.pan_number ? "Yes" : "No"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Format Valid:</span>
                    <span className="text-sm font-medium">
                      {result.valid ? "Yes" : "No"}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

         

          <div className="flex justify-center space-x-4">
            <Button onClick={() => navigate('/pan-verification')}>
              Process Another PAN Card
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

export default PanResults;
