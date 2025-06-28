
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Upload, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const AadhaarVerification = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast({
        title: "Error",
        description: "Please select a file to upload",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/extract-aadhaar`, {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      
      if (response.ok) {
        navigate('/aadhaar-results', { state: { result, fileName: file.name } });
      } else {
        toast({
          title: "Error",
          description: result.error || "Failed to process Aadhaar card",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to connect to server",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100">
      <div className="container mx-auto px-4 py-8">
        <Button 
          onClick={() => navigate('/')} 
          variant="ghost" 
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Button>

        <div className="max-w-2xl mx-auto">
          <Card className="shadow-lg border-0">
            <CardHeader className="text-center pb-6">
              <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-2xl font-bold">Aadhaar Card Verification</CardTitle>
              <CardDescription>
                Upload your Aadhaar card image for automatic text extraction and verification
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Upload Aadhaar Card Image</label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-green-400 transition-colors">
                    <Upload className="w-8 h-8 mx-auto mb-4 text-gray-400" />
                    <Input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                      id="aadhaar-upload"
                    />
                    <label 
                      htmlFor="aadhaar-upload" 
                      className="cursor-pointer text-green-600 hover:text-green-700"
                    >
                      {file ? file.name : "Click to select Aadhaar card image"}
                    </label>
                    <p className="text-xs text-gray-500 mt-2">
                      Supports PNG, JPG, JPEG formats
                    </p>
                  </div>
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-green-500 hover:bg-green-600" 
                  disabled={!file || loading}
                >
                  {loading ? "Processing..." : "Extract Aadhaar Details"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default AadhaarVerification;
