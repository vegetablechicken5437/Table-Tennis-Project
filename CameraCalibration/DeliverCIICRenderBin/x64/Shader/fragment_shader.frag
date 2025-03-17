// Inputs
varying vec4 pos;


float distance;  // Distance between light position and fragment
float NdotL;     // Dot product of normal and lightDir
float attenuation;

vec4 finalcolor;
vec4 diffuse = vec4(2.0,2.0,2.0,1.0);

vec3 normal;
vec3 lightDir;

uniform sampler2D Texture1;			    // Our Texture no.1
uniform sampler2D NormalMap;			// Our Normal Map

void main( void )
{
	//Extract normal from the NormalMap
	//R=x, G=y, B=z
	vec3 NormalTex = texture2D(NormalMap, gl_TexCoord[0].st).xyz;
   
	//Bring the xyz normal in the -1.0 to 1.0 margin
	NormalTex = (NormalTex - 0.5) * 2.0;
		
	//Finally assign the NormalTex to the actual normal
	//to be used in our lighting
	normal=normalize(NormalTex);

	// Compute Lighting
	lightDir = normalize(vec3(gl_LightSource[0].position-pos));
	distance = length(vec3(gl_LightSource[0].position-pos));
	NdotL = max(dot(normal,lightDir),0.0);
	attenuation = 1.00 / (gl_LightSource[0].constantAttenuation +
					gl_LightSource[0].linearAttenuation * distance +
					gl_LightSource[0].quadraticAttenuation * distance * distance);

	// Add texel color
	finalcolor = texture2D(Texture1,gl_TexCoord[0].xy);  

	// Put all the lighting together
	finalcolor *= attenuation * (diffuse * NdotL); 

	gl_FragColor = finalcolor; 
}
